import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from utils import Optimer, compare_model_grad, compare_model_weight
from torch.optim import AdamW, Adam

# import torch_tpu
torch.manual_seed(1000)
# optimer = Optimer("../../build/torch_tpu/libtorch_tpu.so")
import torch_tpu

def case_forward(use_fp16=False):
    device = "tpu:0"
    batch = 8
    sequence = 1024
    hidden_size = 768
    out_size = 3

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp_cpu.to(device).to(torch.float16) #.half()
    ln_cpu = nn.Linear(hidden_size, out_size)
    ln_tpu = copy.deepcopy(ln_cpu)
    ln_tpu = ln_tpu.to(device)
    ln_tpu.to(torch.float16)
    import pdb;pdb.set_trace()

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
    device = "tpu"
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

def case_update_weight(use_fp16=False):
    device = "tpu"
    batch = 32
    sequence = 256
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu)
    inp_cpu.requires_grad = True
    inp_tpu = inp_tpu.to(device)
    inp_tpu.requires_grad = True
    if use_fp16 : inp_tpu = inp_tpu.half()

    grad_cpu = torch.rand(batch, sequence, hidden_size*3)
    grad_tpu = grad_cpu.to(device) 
    if use_fp16:  grad_tpu = grad_tpu.half()

    ln_cpu = nn.Linear(hidden_size, 3 * hidden_size)
    ln_tpu = copy.deepcopy(ln_cpu)
    ln_tpu = ln_tpu.to(device) #.half()
    if use_fp16: ln_tpu = ln_tpu.half()
    ln_cpu.train()
    ln_tpu.train()

    optimizer_cpu = Adam(ln_cpu.parameters(), lr = 1)
    optimizer_tpu = Adam(ln_tpu.parameters(), lr = 1)
    optimizer_cpu.zero_grad()
    optimizer_tpu.zero_grad()

    print("=====forward======")
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

    compare_model_grad(ln_cpu, ln_tpu)
    compare_model_weight(ln_cpu, ln_tpu)

    print("=====update parameters======")
    t1 = time.time()
    optimizer_cpu.step()
    t2 = time.time()
    print("cpu time ",t2 -t1)

    t1 = time.time()
    optimizer_tpu.step()
    t2 = time.time()
    print("tpu time ",t2 -t1)

    compare_model_weight(ln_cpu, ln_tpu)

    # print("============compare grad=========")
    # diff = torch.max(abs(inp_cpu.grad - inp_tpu.grad.cpu()))
    # print ("diff:", torch.max(abs(diff)))

if __name__ == "__main__":
    case_forward(True)
    #case_backward(True)
    #case_update_weight()
