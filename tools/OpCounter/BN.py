from .Utils import *

def get_bytes(shape: list):
    if len(shape) == 0: return 0
    bytes = 4 # f32
    for s in shape:
        bytes *= s
    return bytes

def get_bn_bytes(I_shape : list):
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    WBytes = I_shape[1] * 4
    BBytes = WBytes
    IBytes = get_bytes(I_shape)
    OBytes = IBytes + WBytes + WBytes # out + running_mean + running_var
    return WBytes + IBytes + BBytes + OBytes

def BNDmaTimeUs(I_shape : list):
    allDMA_Bytes = get_bn_bytes(I_shape)
    time_us = (allDMA_Bytes / GB) / BW * US
    return time_us 

def get_bn_flops(I_shape : list):
    # y = (x - mean) / sqrt(eps + var) * weight + bias
    numel = get_bytes(I_shape)/4
    mean_flops = numel              # m1   = x.mean(0).mean(-1).mean(-2)
    var_flops  = 3 * numel          # var1 = (x - m1)**2).mean(0).mean(-1).mean(-2)
    sqrt_flops = 2 * I_shape[1]     # tmp1 = sqrt(eps + var1)
    w_flops    = 4 * numel          # (x - m1)/tmp1  * weight + bias
    flops = mean_flops + var_flops + sqrt_flops + w_flops
    return flops

def BNTiuTimeUs(I_shape : list):
    flops =get_bn_flops(I_shape)
    return flops / F32_FLOPS * US


""" BatchNorm Backward"""

def get_bn_bwd_bytes(I_shape: list, save_mean_var: bool = True):
    """
    // Input Normalized = ( Input − Saved Mean ) × Saved Invstd
    // Grad Weight = ∑<n, h, w> Grad Output × Input Normalized
    // Grad Bias = ∑<n, h, w> Grad Output
    // Grad Input = Weight × Saved Invstd × ( Grad Output − ( Input Normalized × Grad Weight + Grad Bias ) / NHW )
    估算 BN backward 涉及的内存传输字节数(粗略上界)。
    约定:
    - I_shape = [N, C, H, W](至少包含维度,C=通道数)
    - 输入/输出:X, dY, dX
    - 参数:weight(γ), bias(β)
    - 缓存:running_mean, running_var(或 saved_mean, saved_invstd)
    - backward 需要读:X, dY, weight, saved_mean, saved_invstd
    - 写:dX, dweight, dbias
    - 若 save_mean_var=True,认为从前向保存了 mean 和 invstd(或 var),需要读取它们。
    """
    assert len(I_shape) >= 2, "I_shape 至少包含 N 和 C 维度"
    C = I_shape[1]

    # 基本大小
    num_bytes = get_bytes(I_shape)    # for X 或 dY 等张量大小
    tensor_size = num_bytes           # NCHW f32 大小
    param_bytes = C * 4               # 单通道参数或统计量所占字节

    # 参与 backward 的张量访问(粗略估计:读写均算一次)
    # 读:X, dY, weight(γ), saved_mean, saved_invstd
    read_X   = tensor_size
    read_dY  = tensor_size
    read_W   = param_bytes
    read_mean     = param_bytes if save_mean_var else 0
    read_invstd   = param_bytes if save_mean_var else 0  # 保存的是 invstd 比保存 var 更常见

    # 写:dX(同大小),dweight(=C), dbias(=C)
    write_dX  = tensor_size
    write_dW  = param_bytes
    write_dB  = param_bytes

    # 另外:有些实现会读 bias/running stats,但 backward 通常不需要读 β,running_mean/var 多半不更新或不参与梯度
    # 如果你的实现会用到,可加上:
    read_B = 0  # param_bytes

    allDMA_Bytes = (read_X + read_dY + read_W + read_mean + read_invstd + read_B +
                    write_dX + write_dW + write_dB)
    return allDMA_Bytes

def BNBackward_DmaTimeUs(I_shape: list, save_mean_var: bool = True):
    allDMA_Bytes = get_bn_bwd_bytes(I_shape, save_mean_var=save_mean_var)
    time_us = (allDMA_Bytes / GB) / BW * US
    return time_us

def get_bn_bwd_flops(I_shape: list):
    """
    估算 BN backward 的 FLOPs(简化版)。
    推导基于常见实现步骤(以 NCHW、按通道归约为例):
    - 令 numel = N*C*H*W,per_channel = N*H*W
    - 已知 saved_mean, saved_invstd,以及 X、dY、weight(γ)
    - 计算:
        dbias = sum(dY)                         -> per_channel adds
        dweight = sum(dY * X_hat)               -> mul + per_channel adds
        dX = (1/per_channel) * invstd * weight * [ per_channel*dY - dbias - X_hat * dweight ]
      为了避免 per-element 再做归约,通常做两次通道内归约得到 dbias, dweight 后,再一次性计算 dX。
    计数(粗略):
      1) 计算 X_hat = (X - mean) * invstd
         - sub: numel
         - mul: numel
      2) 归约:
         - dbias = sum(dY): numel-1 次加法 ~ numel
         - dweight = sum(dY * X_hat):
              mul: numel
              add: numel
      3) 计算 dX:
         per element:
           t = per_channel * dY - dbias - X_hat * dweight
             - mul: 1 (per_channel * dY,可预合并为标量再乘,简化处理按 mul 1 次)
             - sub/add: 2
             - mul: 1 (X_hat * dweight)
           dX = (1/per_channel) * invstd * weight * t
             - mul: 3
         合计 per element 约: mul 1(上) + mul1(上) + mul3 = 5 mul; 加减约 2
         简化取:每元素 ~ 5 mul + 2 add ≈ 7 FLOPs
    汇总:
      - X_hat: 2*numel
      - dbias: ~ numel
      - dweight: mul numel + add numel = 2*numel
      - dX: ~ 7*numel
      合计基础项:2 + 1 + 2 + 7 = 12*numel FLOPs
      - 少量通道级别操作(比如计算常量系数)忽略,或加上 O(C) 项。
    """
    if len(I_shape) == 0:
        return 0
    numel = get_bytes(I_shape) // 4
    C = I_shape[1] if len(I_shape) >= 2 else 1

    base_flops = 12 * numel
    channel_flops = 10 * C  # 预留一些通道级别标量操作(非常小,可忽略)
    return base_flops + channel_flops

def BNBackward_TiuTimeUs(I_shape: list):
    flops = get_bn_bwd_flops(I_shape)
    return flops / F32_FLOPS * US