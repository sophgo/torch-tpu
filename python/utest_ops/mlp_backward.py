import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
import pdb
import logging
from loguru import logger
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

def check_mlp():

    batch_size = 1  #1
    seq_len = 4096 #128000  #8000
    embed_dim = 3584 #8192 #8192
    intermediate_size = 9427 #29696  #29696  单芯
    custom_use_cpu = True
    return_mid_tensor = False


    log_file = "compare.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    logger.add(log_file, level="DEBUG", format="{time} {level} {message}")

    net_cpu = LLamaMlp(embed_dim, intermediate_size)

    w0_tpu_cpu = copy.deepcopy(net_cpu.mm0.weight.detach()).detach().requires_grad_(True).to(device).half()
    w1_tpu_cpu = copy.deepcopy(net_cpu.mm1.weight.detach()).detach().requires_grad_(True).to(device).half()
    w2_tpu_cpu = copy.deepcopy(net_cpu.mm2.weight.detach()).detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    w0_tpu_cpu.retain_grad()
    w1_tpu_cpu.retain_grad()
    w2_tpu_cpu.retain_grad()


    w0_tpu_tpu = copy.deepcopy(net_cpu.mm0.weight.detach()).detach().requires_grad_(True).to(device).half()
    w1_tpu_tpu = copy.deepcopy(net_cpu.mm1.weight.detach()).detach().requires_grad_(True).to(device).half()
    w2_tpu_tpu = copy.deepcopy(net_cpu.mm2.weight.detach()).detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    w0_tpu_tpu.retain_grad()
    w1_tpu_tpu.retain_grad()
    w2_tpu_tpu.retain_grad()

    net_tpu_cpu = LLamaMlpBlock(w0_tpu_cpu, w1_tpu_cpu, w2_tpu_cpu, True, return_mid_tensor, True)
    net_tpu_tpu = LLamaMlpBlock(w0_tpu_tpu, w1_tpu_tpu, w2_tpu_tpu, True, return_mid_tensor, False)

    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)*0.1
    #x = torch.ones(batch_size, seq_len, embed_dim, requires_grad=True) * 0.1

    x_tpu_cpu = copy.deepcopy(x.detach()).requires_grad_(True).to(device).half()
    x_tpu_tpu = copy.deepcopy(x.detach()).requires_grad_(True).to(device).half()
    x_tpu_cpu.retain_grad()
    x_tpu_tpu.retain_grad()

    logger.info(f"before of nw bw0")
    out_tpu_cpu = net_tpu_cpu(x_tpu_cpu)
    out_tpu_tpu = net_tpu_tpu(x_tpu_tpu)
    logger.info(f"end of nw bw0")

    print("======backward=======")
    # cpu:backward

    loss_tpu_cpu = out_tpu_cpu.sum()
    loss_tpu_cpu.backward()
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
    comparator = TensorComparator()
    logger.info(f"backward: compare with custom and 1686")
    com_x  = comparator.cmp_result(x_tpu_cpu.grad.cpu().float(), x_tpu_tpu.grad.cpu().float(),"grad_input")
    com_w0 = comparator.cmp_result(grad_tpuc_w0.cpu(), grad_tpu_w0.cpu(),"grad_w0")
    com_w1 = comparator.cmp_result(grad_tpuc_w1.cpu(), grad_tpu_w1.cpu(),"grad_w1")
    com_w2 = comparator.cmp_result(grad_tpuc_w2.cpu(), grad_tpu_w2.cpu(),"grad_w2")
    logger.info(f"x={com_x},w0={com_w0},w1={com_w1},w2={com_w2}")
   # logger.info(f"w1cpu={grad_tpuc_w1.cpu().float()}")
   # logger.info(f"w1tpu={grad_tpu_w1.cpu().float()}")
   # logger.info(f"xc={x_tpu_cpu.grad.cpu().float()}")
    #logger.info(f"xt={x_tpu_tpu.grad.cpu().float()}")


    logger.remove()
    return


if __name__ == "__main__":
    check_mlp()
