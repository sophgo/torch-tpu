import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def cos_sim(vector_a, vector_b):
    vector_a = vector_a.reshape(-1)
    vector_b = vector_b.reshape(-1)
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    with np.errstate(invalid='ignore'):
        cos = np.nan_to_num(num / denom)
    sim = 0.5 + 0.5 * cos
    return sim

def weight_dequant_vector(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor: #faster
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size for tiling. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    m, n = [(i + block_size - 1) // block_size for i in x.size()]

    pad = [0 for i in range(4)]
    if N % block_size:
        pad[1] = block_size - N % block_size
    if M % block_size:
        pad[3] = block_size - M % block_size

    if pad:
        x = F.pad(x, pad)

    bf16_x = x.view(m, block_size, n, block_size).permute(1, 3, 0, 2).to(torch.bfloat16)
    out = (bf16_x * s).permute(2, 0, 3, 1)

    if pad:
        return out.reshape(m * block_size, n * block_size)[:M, :N]
    else:
        return out.reshape(M, N)

def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size for tiling. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())  # 输出张量

    # 模拟 Triton 的分块计算逻辑
    for pid_m in range((M + block_size - 1) // block_size):
        for pid_n in range((N + block_size - 1) // block_size):
            # 计算当前块的起始和结束索引
            offs_m_start = pid_m * block_size
            offs_m_end = min(offs_m_start + block_size, M)
            offs_n_start = pid_n * block_size
            offs_n_end = min(offs_n_start + block_size, N)

            # 提取当前块的量化权重和缩放因子
            x_block = x[offs_m_start:offs_m_end, offs_n_start:offs_n_end]
            s_block = s[pid_m, pid_n]  # 假设 s 的每个块对应一个缩放因子

            # 反量化逻辑：将量化权重乘以缩放因子
            y_block = x_block.to(torch.float32) * s_block
            # print(s_block.dtype)

            # 将结果写回输出张量
            y[offs_m_start:offs_m_end, offs_n_start:offs_n_end] = y_block

    return y

class DeepseekMlp(nn.Module):
    def __init__(self, w0, w1, w2, s0, s1, s2, blocksize):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.blocksize = blocksize

    def forward(self, x):
        dequantized_weight0 = weight_dequant(self.w0, self.s0, self.blocksize)
        out00 = F.linear(x, dequantized_weight0)
        out0 = F.silu(out00)

        dequantized_weight1 = weight_dequant(self.w1, self.s1, self.blocksize)
        out1 = F.linear(x, dequantized_weight1)
        out2 = out0 * out1

        dequantized_weight2 = weight_dequant(self.w2, self.s2, self.blocksize)
        out3 = F.linear(out2, dequantized_weight2)

        return out3

class DeepseekMlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2, s0, s1, s2, blocksize):
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)

        torch.ops.my_ops.mlp_w8a16_quant_forward(x,
                                    w0,
                                    w1,
                                    w2,
                                    s0,
                                    s1,
                                    s2,
                                    output,
                                    blocksize)
        return output

class DeepseekMlpBlock(nn.Module):
    def __init__(self, w0, w1, w2, s0, s1, s2, blocksize):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.blocksize = blocksize

    def forward(self, x):
        return DeepseekMlpFunc.apply(x, self.w0, self.w1, self.w2, self.s0, self.s1, self.s2, self.blocksize)



def check_mlp():
    batch_size = 1024 #4096
    input_w = 7168
    middle_w = 256 #512#4096#4608
    blocksize = 128
    scale_m = (middle_w + blocksize - 1) // blocksize
    scale_n = (input_w + blocksize - 1) // blocksize
    torch.set_default_dtype(torch.bfloat16)

    x = torch.randn(batch_size, input_w) # 输入数据 (batch_size=10, in_features=5)
    w0 = torch.randn(middle_w, input_w) # 量化权重 (out_features=8, in_features=5) torch.float8_e4m3fn
    w1 = torch.randn(middle_w, input_w) # 缩放因子矩阵
    w2 = torch.randn(input_w, middle_w) # 量化权重 (out_features=8, in_features=5)
    s0 = torch.rand(scale_m, scale_n)  # 缩放因子矩阵
    s1 = torch.rand(scale_m, scale_n)  # 量化权重 (out_features=8, in_features=5)
    s2 = torch.rand(scale_n, scale_m)  # 缩放因子矩阵
    # s0 = torch.ones(scale_m, scale_n)  # 缩放因子矩阵
    # s1 = torch.ones(scale_m, scale_n)  # 量化权重 (out_features=8, in_features=5)
    # s2 = torch.ones(scale_n, scale_m)  # 缩放因子矩阵
    w0 = w0.to(torch.float8_e4m3fn)
    w1 = w1.to(torch.float8_e4m3fn)
    w2 = w2.to(torch.float8_e4m3fn)

    net_cpu = DeepseekMlp(w0, w1, w2, s0, s1, s2, blocksize)

    x_tpu = x.to(device)
    w0_tpu = w0.to(device)
    w1_tpu = w1.to(device)
    w2_tpu = w2.to(device)
    s0_tpu = s0.to(device)
    s1_tpu = s1.to(device)
    s2_tpu = s2.to(device)
    # import pdb
    # pdb.set_trace()
    net_tpu = DeepseekMlpBlock(w0_tpu, w1_tpu, w2_tpu, s0_tpu, s1_tpu, s2_tpu, blocksize)

    print("=====forward======")
    out_cpu = net_cpu(x)
    print("=====tpu forward======")
    out_tpu = net_tpu(x_tpu)
    # out_cpu = net_cpu(x)
    # import pdb
    # pdb.set_trace()
    out_cpu = out_cpu.float().flatten()
    out_tpu = out_tpu.float().to("cpu").flatten()
    out_diff = out_cpu - out_tpu
    ratio = abs(out_diff/torch.max(abs(out_cpu), abs(out_tpu)))
    key = torch.argmax(ratio)
    print(out_tpu[key], out_cpu[key], ratio[key])
    cosm = cos_sim(out_cpu.numpy(), out_tpu.numpy())
    print(cosm)
    #print(out_cpu)
    #print(out_tpu)
    return


if __name__ == "__main__":
    check_mlp()
