import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
from top_utest import TensorComparator
from torch_tpu.tpu.custom_op.llama_mlp import LLamaMlpFunc

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
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        r_mm0 = self.mm0(x)
        r_mm1 = self.mm1(x)
        r_mm1 = r_mm1 * self.sigmoid(r_mm1)
        r_tmp = r_mm0 * r_mm1
        x = self.mm2(r_tmp)
        return x

class LLamaMlpBlock(nn.Module):
    def __init__(self, w0, w1, w2):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

    def forward(self, x):
        return LLamaMlpFunc.apply(x, self.w0, self.w1, self.w2)


def check_mlp():
    batch_size = 3
    seq_len = 128
    embed_dim = 256
    intermediate_size = 512
    net_cpu = LLamaMlp(embed_dim, intermediate_size)

    w0_tpu = copy.deepcopy(net_cpu.mm0.weight.detach()).detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    w1_tpu = copy.deepcopy(net_cpu.mm1.weight.detach()).detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    w2_tpu = copy.deepcopy(net_cpu.mm2.weight.detach()).detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    w2_tpu.retain_grad()
    w0_tpu.retain_grad()
    w1_tpu.retain_grad()

    net_tpu = LLamaMlpBlock(w0_tpu, w1_tpu, w2_tpu)

    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)    
    x_tpu = copy.deepcopy(x.detach()).to(device).requires_grad_(True).half()
    x_tpu.retain_grad()

    out_tpu = net_tpu(x_tpu)
    out_cpu = net_cpu(x)

    loss_cpu = out_cpu.sum()
    loss_cpu.backward()
    loss_tpu = out_tpu.sum()
    loss_tpu.backward()
    comparator = TensorComparator()
    status_fwd = comparator.cmp_result(out_cpu.detach(), out_tpu.cpu().detach().float())
    status_bwd_x = comparator.cmp_result(x.grad.detach(), x_tpu.grad.detach().cpu().float())
    status_bwd_w0 = comparator.cmp_result(net_cpu.mm0.weight.grad.detach(), net_tpu.w0.grad.cpu().detach().t().contiguous().float())
    status_bwd_w1 = comparator.cmp_result(net_cpu.mm1.weight.grad.detach(), net_tpu.w1.grad.cpu().detach().t().contiguous().float())
    status_bwd_w2 = comparator.cmp_result(net_cpu.mm2.weight.grad.detach(), net_tpu.w2.grad.cpu().detach().t().contiguous().float())

    status_bwd = status_bwd_x and status_bwd_w0 and status_bwd_w1 and status_bwd_w2
    return status_fwd, status_bwd

if __name__ == "__main__":
    status_fwd, status_bwd = check_mlp()
    if (status_bwd and status_fwd):
        print(f"[Success] llama_mlp compare succeed!")
    else:
        print(f"[Failed] llama_mlp compare failed!")
        sys.exit(255)
