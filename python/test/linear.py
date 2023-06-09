import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from utils import Optimer, compare_model_grad

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
optimer = Optimer("../../libtorch_plugin/build/liblibtorch_plugin.so")

def case_forward(use_fp16=False):
    device = "privateuseone"
    batch = 8
    sequence = 1024
    hidden_size = 768
    out_size = 3

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp_cpu.to(device).half()

    ln_cpu = nn.Linear(hidden_size, out_size)
    ln_tpu = copy.deepcopy(ln_cpu)
    ln_tpu = ln_tpu.to(device).half()

    out_cpu = ln_cpu(inp_cpu)
    out_tpu = ln_tpu(inp_tpu)
    out_tpu = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu
    print("cpu_out")
    print(out_cpu)
    print("tpu_out")
    print(out_tpu)
    
    print (torch.max(abs(out_diff)))

def case_backward(use_fp16=False):
    device = "privateuseone"
    batch = 32
    sequence = 256
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu)
    inp_cpu.requires_grad = True
    inp_tpu = inp_tpu.to(device).half()
    inp_tpu.requires_grad = True

    grad_cpu = torch.rand(batch, sequence, hidden_size*3)
    grad_tpu = grad_cpu.to(device).half()

    ln_cpu = nn.Linear(hidden_size, 3 * hidden_size)
    ln_tpu = copy.deepcopy(ln_cpu)
    ln_tpu = ln_tpu.to(device).half()

    out_cpu = ln_cpu(inp_cpu)
    out_tpu = ln_tpu(inp_tpu)
    out_diff = out_cpu - out_tpu.float().to("cpu")
    print (torch.max(abs(out_diff)))

    print("=====backward======")
    t1 = time.time()
    out_cpu.backward(grad_cpu)
    t2 = time.time()
    print("cpu time ",t2 -t1)

    t1 = time.time()
    out_tpu.backward(grad_tpu)
    t2 = time.time()
    print("tpu time ",t2 -t1)

    print("============compare grad=========")
    diff = torch.max(abs(inp_cpu.grad - inp_tpu.grad.cpu()))
    print ("diff:", torch.max(abs(diff)))

    compare_model_grad(ln_cpu, ln_tpu)

if __name__ == "__main__":
    case_backward(True)
