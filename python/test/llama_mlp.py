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

class LLamaMlp(nn.Module):
    def __init__(self, embed_dim, intermediate_size):
        super().__init__()
        self.mm0 = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.mm1 = nn.Linear(embed_dim, intermediate_size, bias=False)
        self.mm2 = nn.Linear(intermediate_size, embed_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r_mm0 = self.mm0(x)
        r_mm1 = self.mm1(x)
        r_mm1 = r_mm1 * self.sigmoid(r_mm1)
        r_tmp = r_mm0 * r_mm1
        x = self.mm2(r_tmp)
        return x


class LLamaMlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2):
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)

        torch.ops.my_ops.llama_mlp_forward(x,
                                    w0,
                                    w1,
                                    w2,
                                    output)
        return output

class LLamaMlpBlock(nn.Module):
    def __init__(self, w0, w1, w2):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

    def forward(self, x):
        return LLamaMlpFunc.apply(x, self.w0, self.w1, self.w2)


def check_mlp():
    batch_size = 1
    embed_dim = 128
    intermediate_size = 256
    net_cpu = LLamaMlp(embed_dim, intermediate_size)

    w0 = net_cpu.state_dict()['mm0.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(False).to(device).half()
    w1 = net_cpu.state_dict()['mm1.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(False).to(device).half()
    w2 = net_cpu.state_dict()['mm2.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(False).to(device).half()

    net_tpu = LLamaMlpBlock(w0, w1, w2)

    print("=====forward======")
    x = torch.randn(batch_size, embed_dim, requires_grad=True)
    x_tpu = x.to(device).half()
    out_tpu = net_tpu(x_tpu)
    out_cpu = net_cpu(x)
    out_diff = out_cpu - out_tpu.float().to("cpu")
    print (torch.max(abs(out_diff)))
    return


if __name__ == "__main__":
    check_mlp()
