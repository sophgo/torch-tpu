import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_tpu
torch.manual_seed(1000)

class NativeMatmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias=None):
        return F.linear(input, weight, bias)

class A16MatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, active, weight, bias, scale, zp, group_size, weight_bits):
        output = torch.empty((active.shape[0], active.shape[1], active.shape[2], weight.shape[2]), dtype = active.dtype, device = weight.device)

        torch.ops.my_ops.matmul_gptq_forward(active,
                                    weight,
                                    bias,
                                    scale,
                                    zp,
                                    group_size,
                                    weight_bits,
                                    output)
        return output

class A16Matmul(nn.Module):
    def __init__(self, scale, zp, group_size, weight_bits):
        super().__init__()
        self.scale = scale
        self.zp = zp
        self.group_size = group_size
        self.weight_bits = weight_bits

    def forward(self, active, weight, bias):
        return A16MatmulFunc.apply(active, weight, bias, self.scale, self.zp, self.group_size, self.weight_bits)

def check_a16_matmul():
    device = "tpu"
    quant_uint4 = True
    scale_zp_zip = True
    has_bias = True
    dtype = torch.float16
    # dtype = torch.bfloat16

    # input: [1, 1, 16, 8192] * [1, 1, 8192, 1280]    8bit--pass   4bit--pass
    # input: [1, 1, 16, 8192] * [1, 1, 8192, 128]     8bit--pass   4bit--pass
    # input: [1, 1, 16, 8192] * [1, 1, 8192, 3584]    8bit--pass   4bit--pass
    # input: [1, 1, 16, 3584] * [1, 1, 3584, 8192]    8bit--pass   4bit--pass
    # input: [1, 1, 128, 4096] * [1, 1, 4096, 128]    8bit--pass   4bit--pass
    # input: [1, 1, 16, 8192] * [1, 1, 8192, 32000]   8bit--pass   4bit--pass
    b = 1
    m = 1
    l = 16
    d = 8192
    n = 1280

    q_cpu = torch.randn(b, m, l, d)
    k_cpu = torch.randn(b, m, n, d)
    bias_cpu = torch.randn(n) if has_bias else None

    scale_t = torch.full([n, d // 128], 0.1, dtype=dtype)
    zp_t = torch.full([n, d // 128], 3, dtype=torch.uint8)
    # zp_t = torch.randint(1, 3, [n, d // 128], dtype=torch.uint8)
    # zp_pad = torch.zeros([n, d], dtype=torch.uint8)
    # for i in range(0,n):
    #     for j in range(0,d//128):
    #         for k in range(0,128):
    #             zp_pad[i][j*128+k] = zp_t[i][j]
    # scale_t = torch.rand((n, d // 128), dtype=torch.float16)
    # scale_pad = torch.zeros([n, d], dtype=torch.float32)
    # for i in range(0,n):
    #     for j in range(0,d//128):
    #         for k in range(0,128):
    #             scale_pad[i][j*128+k] = scale_t[i][j]

    k_cpu_qt = torch.quantize_per_tensor(k_cpu, 0.1, 3, torch.qint8)
    k_cpu_qt = k_cpu_qt.int_repr()
    if quant_uint4:
        k_int8_tensor = torch.randint(4, 15, (n, d), dtype=torch.uint8)
        k_cpu_qt = torch.bitwise_or(k_int8_tensor.transpose(-1,-2)[::2], k_int8_tensor.transpose(-1,-2)[1::2] << 4).transpose(-1,-2)
        k_cpu_qt = torch.unsqueeze(torch.unsqueeze(k_cpu_qt,0),0)
    # kdq = k_cpu_qt.dequantize()
    print(q_cpu.half())
    if (quant_uint4):
        print(k_int8_tensor)
    print(k_cpu_qt)
    if (has_bias):
        print(bias_cpu.half())

    if quant_uint4:
        net_cpu = NativeMatmul()
        k_deq = (k_int8_tensor - 3) * 0.1
        out_cpu = net_cpu(q_cpu[0][0], k_deq, bias_cpu)
    else:
        net_cpu = NativeMatmul()
        k_deq = ((k_cpu_qt - 3) * 0.1).reshape(-1, d)
        out_cpu = net_cpu(q_cpu[0][0], k_deq, bias_cpu)

    weight_bits = 8
    if quant_uint4:
        zp_t = torch.bitwise_or(zp_t.transpose(-1,-2)[::2], zp_t.transpose(-1,-2)[1::2] << 4).transpose(-1,-2)
        weight_bits = 4
    if scale_zp_zip:
        scale_t = torch.cat((scale_t.view(dtype=torch.uint8), zp_t), axis=-1)


    net_tpu = A16Matmul(scale_t.to(device), zp_t.to(device), 128, weight_bits)
    if (has_bias):
        out_tpu = net_tpu(q_cpu.to(dtype).to(device), k_cpu_qt.to(device), bias_cpu.to(dtype).to(device)).cpu()
    else:
        out_tpu = net_tpu(q_cpu.to(dtype).to(device), k_cpu_qt.to(device), bias_cpu).cpu()

    print(out_cpu.shape, out_tpu.shape)
    print(out_cpu)
    print(out_tpu[0][0])
    print("max diff:",torch.max(abs(out_cpu - out_tpu.cpu())))




if __name__ == "__main__":
    check_a16_matmul()
