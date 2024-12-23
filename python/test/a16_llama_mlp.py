import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
from safetensors.torch import load_file
import math

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
np.set_printoptions(threshold=np.inf, precision=3,suppress=True)

hex_dunc = np.vectorize(hex)
import time
class LLamaA16Mlp(nn.Module):
    def __init__(self, weight0_quant, weight1_quant, weight2_quant, q_group_size, weight_bits):
        super().__init__()
        self.q_group_size = q_group_size
        self.weight_bits = weight_bits
        self.sigmoid = nn.Sigmoid()
        #dequant weight
        self.w0 = self.dequant(weight0_quant).float()
        self.w1 = self.dequant(weight1_quant).float()
        self.w2 = self.dequant(weight2_quant).float()

    def dequant(self, quantized_weights):
        qweight = quantized_weights["weight"]
        qzeros = quantized_weights["zp"]
        qscale = quantized_weights["scale"]
        i = 0
        col_ = 0
        weight_split = torch.empty((qweight.shape[0], qweight.shape[1] * 8 // self.weight_bits), dtype=torch.uint8)
        while col_ < qweight.shape[1]:
            for j in range(i, i + (8 // self.weight_bits)):
                weight_split[:, j] = (qweight[:, col_] >> self.weight_bits * (j - i)) & 0xF
            i += 8 // self.weight_bits
            col_ += 1
        # split z
        i = 0
        col_ = 0
        zeros_split = torch.empty((qzeros.shape[0], qzeros.shape[1] * 8 // self.weight_bits), dtype=torch.uint8)
        while col_ < qzeros.shape[1]:
            for j in range(i, i + (8 // self.weight_bits)):
                zeros_split[:, j] = (qzeros[:, col_] >> self.weight_bits * (j - i)) & 0xF
            i += 8 // self.weight_bits
            col_ += 1
        # w = w - z
        zeros = torch.empty((zeros_split.shape[0], zeros_split.shape[1] * self.q_group_size), dtype=torch.int8)
        for i in range(zeros_split.shape[1]):
            zeros[:, i*self.q_group_size:(i+1)*self.q_group_size] = zeros_split[:, i:i+1]
        zeros = zeros[:weight_split.shape[0],:weight_split.shape[1]]
        dequant_weight = weight_split - zeros
        # dequant_weigth = w * s
        scale = torch.empty((qscale.shape[0], qscale.shape[1] * self.q_group_size), dtype=torch.float16)
        for i in range(qscale.shape[1]):
            scale[:, i*self.q_group_size:(i+1)*self.q_group_size] = qscale[:, i:i+1]
        dequant_weight = dequant_weight * scale
        return dequant_weight

    def forward(self, x):
        r_mm0 = F.linear(x, self.w0)
        r_mm1 = F.linear(x, self.w1)
        r_mm1 = r_mm1 * self.sigmoid(r_mm1)
        r_tmp = r_mm0 * r_mm1
        x = F.linear(r_tmp, self.w2)
        return x

class LLamaA16MlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight0, weight1, weight2, group_size, weight_bits):
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)
        w0 = weight0["weight"]
        z0 = weight0["zp"]
        s0 = weight0["scale"]

        w1 = weight1["weight"]
        z1 = weight1["zp"]
        s1 = weight1["scale"]

        w2 = weight2["weight"]
        z2 = weight2["zp"]
        s2 = weight2["scale"]
        LOOP = 1000
        t0 = time.time_ns()
        torch.ops.my_ops.llama_mlp_gptq_forward(x, w0, z0, s0, w1, z1, s1, w2, z2, s2, group_size, weight_bits, output)
        t1 = time.time_ns()
        for i in range(LOOP):
            torch.ops.my_ops.llama_mlp_gptq_forward(x, w0, z0, s0, w1, z1, s1, w2, z2, s2, group_size, weight_bits, output)
        torch_tpu.tpu.synchronize()
        t2 = time.time_ns()
        print(f" warmup                time = {(t1 - t0)/(1e6)} ms")
        print(f" loop {LOOP}, per loop time = {(t2 - t1)/(1e6 * LOOP)} ms")
        return output

class LLamaA16MlpBlock(nn.Module):
    def __init__(self, w0, w1, w2, group_size = 128, weight_bits = 4):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.group_size = group_size
        self.weight_bits = weight_bits

    def forward(self, x):
        return LLamaA16MlpFunc.apply(x, self.w0, self.w1, self.w2, self.group_size, self.weight_bits)

def check_mlp_a16(model="qwen72b",batch=16,tp=2,seq_len=5):

    batch_size = batch*seq_len
    if model=="qwen72b":
        embed_dim = 8192
        intermediate_size = (int)(29696/tp)
    elif model=="qwen7b":
        embed_dim = 3584
        intermediate_size = (int)(18944/tp)
    elif model=="llama7b":
        embed_dim = 4096
        intermediate_size = (int)(11008/tp)

    group_size = 128
    weight_bits = 4
    zp_value = 119
    weight0_quant = {} #up_proj
    weight0_quant["weight"] = torch.randint(0, 255, (int(intermediate_size), int(embed_dim / 8 * weight_bits)), dtype=torch.uint8)
    weight0_quant["zp"] = torch.full((int(intermediate_size), int(embed_dim / group_size / 8 * weight_bits)), zp_value, dtype=torch.uint8)
    weight0_quant["scale"] = (torch.rand((int(intermediate_size), int(embed_dim / group_size)), dtype=torch.float32)*(0.005-0.002)+0.002).half()
    weight0_quant_tpu = {}
    weight0_quant_tpu["weight"] = weight0_quant["weight"].to(device)
    weight0_quant_tpu["zp"] = weight0_quant["zp"].to(device)
    weight0_quant_tpu["scale"] = weight0_quant["scale"].to(device)

    weight1_quant = {} #gate_proj
    weight1_quant["weight"] = torch.randint(0, 255, (int(intermediate_size), int(embed_dim / 8 * weight_bits)), dtype=torch.uint8)
    weight1_quant["zp"] = torch.full((int(intermediate_size), int(embed_dim / group_size / 8 * weight_bits)), zp_value, dtype=torch.uint8)
    weight1_quant["scale"] = (torch.rand((int(intermediate_size), int(embed_dim / group_size)), dtype=torch.float32)*(0.005-0.002)+0.002).half()
    weight1_quant_tpu = {}
    weight1_quant_tpu["weight"] = weight1_quant["weight"].to(device)
    weight1_quant_tpu["zp"] = weight1_quant["zp"].to(device)
    weight1_quant_tpu["scale"] = weight1_quant["scale"].to(device)

    weight2_quant = {} #down_proj
    weight2_quant["weight"] = torch.randint(0, 255, (int(embed_dim), int(intermediate_size / 8 * weight_bits)), dtype=torch.uint8)
    weight2_quant["zp"] = torch.full((int(embed_dim), int(math.ceil(intermediate_size / group_size / 8 * weight_bits))), zp_value, dtype=torch.uint8)
    weight2_quant["scale"] = (torch.rand((int(embed_dim), int(intermediate_size / group_size)), dtype=torch.float32)*(0.005-0.002)+0.002).half()
    weight2_quant_tpu = {}
    weight2_quant_tpu["weight"] = weight2_quant["weight"].t().view(-1,group_size,embed_dim).permute(0,2,1).contiguous().view(-1,group_size*embed_dim).to(device)
    weight2_quant_tpu["zp"] = weight2_quant["zp"].t().to(device)
    weight2_quant_tpu["scale"] = weight2_quant["scale"].t().to(device)

    net_cpu = LLamaA16Mlp(weight0_quant, weight1_quant, weight2_quant, group_size, weight_bits)
    net_tpu = LLamaA16MlpBlock(weight0_quant_tpu, weight1_quant_tpu, weight2_quant_tpu, group_size, weight_bits)

    x = torch.randn(batch_size, embed_dim, requires_grad=True)
    x_tpu = x.half().to(device)
    out_cpu = net_cpu(x)
    out_tpu = net_tpu(x_tpu)
    out_diff = out_cpu - out_tpu.to("cpu").float()
    print ("result",torch.max(abs(out_diff)))
    return


if __name__ == "__main__":
    check_mlp_a16()
