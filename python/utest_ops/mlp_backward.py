import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
import pdb
import os
import torch_tpu
from python.utest_ops.top_utest import TensorComparator
from python.utest_ops.llama_mlp import LLamaMlp
from torch_tpu.tpu.custom_op.llama_mlp import LLamaMlpFunc

seed=1000
torch.manual_seed(seed)
device = "tpu:0"

class LLamaMlpBlock(nn.Module):
    def __init__(self, w0_tpu_cpu, w1_tpu_cpu, w2_tpu_cpu, use_cpu_fw=False, return_mid_tensor=False, use_cpu_bw=False):
        super().__init__()
        self.w0_tpu_cpu = w0_tpu_cpu
        self.w1_tpu_cpu = w1_tpu_cpu
        self.w2_tpu_cpu = w2_tpu_cpu
        self.use_cpu_fw = use_cpu_fw
        self.use_cpu_bw = use_cpu_bw
        self.return_mid_tensor = return_mid_tensor

    def forward(self, x):
        return LLamaMlpFunc.apply(x, self.w0_tpu_cpu, self.w1_tpu_cpu, self.w2_tpu_cpu, self.use_cpu_fw, self.return_mid_tensor, self.use_cpu_bw)

def check_mlp(dtype: torch.dtype):

    batch_size = 1  #1
    seq_len = 4096  #128000
    embed_dim = 3584  #8192
    intermediate_size = 9427  #29696
    custom_use_cpu = True
    return_mid_tensor = False

    net_cpu = LLamaMlp(embed_dim, intermediate_size)

    w0_tpu_cpu = copy.deepcopy(net_cpu.mm0.weight.detach()).detach().requires_grad_(True).to(device).to(dtype)
    w1_tpu_cpu = copy.deepcopy(net_cpu.mm1.weight.detach()).detach().requires_grad_(True).to(device).to(dtype)
    w2_tpu_cpu = copy.deepcopy(net_cpu.mm2.weight.detach()).detach().transpose(0,1).contiguous().requires_grad_(True).to(device).to(dtype)
    w0_tpu_cpu.retain_grad()
    w1_tpu_cpu.retain_grad()
    w2_tpu_cpu.retain_grad()


    w0_tpu_tpu = copy.deepcopy(net_cpu.mm0.weight.detach()).detach().requires_grad_(True).to(device).to(dtype)
    w1_tpu_tpu = copy.deepcopy(net_cpu.mm1.weight.detach()).detach().requires_grad_(True).to(device).to(dtype)
    w2_tpu_tpu = copy.deepcopy(net_cpu.mm2.weight.detach()).detach().transpose(0,1).contiguous().requires_grad_(True).to(device).to(dtype)
    w0_tpu_tpu.retain_grad()
    w1_tpu_tpu.retain_grad()
    w2_tpu_tpu.retain_grad()

    net_tpu_cpu = LLamaMlpBlock(w0_tpu_cpu, w1_tpu_cpu, w2_tpu_cpu, False, return_mid_tensor, True)
    net_tpu_tpu = LLamaMlpBlock(w0_tpu_tpu, w1_tpu_tpu, w2_tpu_tpu, False, return_mid_tensor, False)
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)*0.1

    x_tpu_cpu = copy.deepcopy(x.detach()).requires_grad_(True).to(device).to(dtype)
    x_tpu_tpu = copy.deepcopy(x.detach()).requires_grad_(True).to(device).to(dtype)
    x_tpu_cpu.retain_grad()
    x_tpu_tpu.retain_grad()

    out_tpu_cpu = net_tpu_cpu(x_tpu_cpu)
    out_tpu_tpu = net_tpu_tpu(x_tpu_tpu)

    print("======backward=======")

    loss_tpu_cpu = out_tpu_cpu.sum()
    # max_record_num = int(1e6)
    # book_keeping = 1
    # torch.ops.my_ops.enable_profile(max_record_num, book_keeping)

    loss_tpu_cpu.backward()
    # torch.ops.my_ops.disable_profile()
    loss_tpu_tpu = out_tpu_tpu.sum()
    # max_record_num = int(1e6)
    # book_keeping = 1
    # torch.ops.my_ops.enable_profile(max_record_num, book_keeping)
    loss_tpu_tpu.backward()
    # torch.ops.my_ops.disable_profile()

    # backward in the tpu
    grad_tpu_w0 = net_tpu_tpu.w0_tpu_cpu.grad.detach().to("cpu").float()
    grad_tpu_w1 = net_tpu_tpu.w1_tpu_cpu.grad.detach().to("cpu").float()
    grad_tpu_w2 = net_tpu_tpu.w2_tpu_cpu.grad.detach().to("cpu").transpose(0,1).contiguous().float()

    grad_tpuc_w0 = net_tpu_cpu.w0_tpu_cpu.grad.detach().to("cpu").float()
    grad_tpuc_w1 = net_tpu_cpu.w1_tpu_cpu.grad.detach().to("cpu").float()
    grad_tpuc_w2 = net_tpu_cpu.w2_tpu_cpu.grad.detach().to("cpu").transpose(0,1).contiguous().float()

   # torch.set_printoptions(threshold=float('inf'))
   # pdb.set_trace()
    print(f"/********************************************************/")
    comparator = TensorComparator()
    com_x  = comparator.cmp_result(x_tpu_cpu.grad.cpu().float(), x_tpu_tpu.grad.cpu().float(),"grad_input")
    com_w0 = comparator.cmp_result(grad_tpuc_w0.cpu(), grad_tpu_w0.cpu(),"grad_w0")
    com_w1 = comparator.cmp_result(grad_tpuc_w1.cpu(), grad_tpu_w1.cpu(),"grad_w1")
    com_w2 = comparator.cmp_result(grad_tpuc_w2.cpu(), grad_tpu_w2.cpu(),"grad_w2")
    print(f"x={com_x},w0={com_w0},w1={com_w1},w2={com_w2}")
    print(f"/********************************************************/")

    com_bwd = com_w0 and com_w1 and com_w2
    if com_bwd and com_x:
        print(f"[Success] llama_mlp_backward compare succeed!")
    else:
        print(f"[Failed] llama_mlp_backward compare failed!")
        sys.exit(255)

    return com_bwd, com_x


if __name__ == "__main__":
    if os.environ['CHIP_ARCH'] in ['bm1684x']:
        print(f'Skip test for this arch')
        sys.exit(0)

    print("==========float16==========")
    check_mlp(torch.float16)
    print("==========bfloat16==========")
    check_mlp(torch.bfloat16)
