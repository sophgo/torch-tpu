import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

def case_forward(use_fp16 = False):
    device = "privateuseone"
    batch = 1
    head_num = 1
    sequence = 4
    i_tmp = []
    for i in range(sequence * sequence):
        i_tmp.append(i)
    inp_cpu = torch.Tensor(i_tmp).view(batch, head_num, sequence, sequence).float()

    inp_tpu = inp_cpu.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    net_cpu = nn.Softmax(-1)
    net_tpu = copy.deepcopy(net_cpu)
    net_tpu = net_tpu.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    t1 = time.time()
    out_cpu = net_cpu(inp_cpu)
    t2 = time.time()
    print("cpu time ",t2 -t1)

    t1 = time.time()
    out_tpu = net_tpu(inp_tpu)
    t2 = time.time()
    print("tpu time ",t2 -t1)

    out_tpu = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu
    print ("diff:", torch.max(abs(out_diff)))

def case_backward(use_fp16 = False):
    device = "privateuseone"
    batch = 32
    head_num = 12
    sequence = 1024

    inp_cpu = torch.rand(batch, head_num, sequence, sequence)
    inp_tpu = copy.deepcopy(inp_cpu)
    inp_tpu = inp_tpu.to(device)
    inp_cpu.requires_grad = True

    grad_ref = torch.rand(batch, head_num, sequence, sequence)
    grad_ref_tpu = grad_ref.to(device)

    if use_fp16: inp_tpu = inp_tpu.half(); grad_ref_tpu.half()
    inp_tpu.requires_grad = True

    net_cpu = nn.Softmax(-1)
    net_tpu = copy.deepcopy(net_cpu)
    net_tpu = net_tpu.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    print("============forward===========")
    t1 = time.time()
    out_cpu = net_cpu(inp_cpu)
    t2 = time.time()
    print("cpu time ",t2 -t1)

    t1 = time.time()
    out_tpu = net_tpu(inp_tpu)
    t2 = time.time()
    print("tpu time ",t2 -t1)
    
    out_diff = out_cpu - out_tpu.float().to("cpu")
    print ("diff:", torch.max(abs(out_diff)))
    print("============backward===========")
    t1 = time.time()
    out_cpu.backward(grad_ref)
    t2 = time.time()
    print("cpu time ",t2 -t1)

    t1 = time.time()
    out_tpu.backward(grad_ref_tpu)
    t2 = time.time()
    print("tpu time ",t2 -t1)

    print("============compare grad=========")
    diff = torch.max(abs(inp_cpu.grad - inp_tpu.grad.cpu()))
    print ("diff:", torch.max(abs(diff)))

if __name__ == "__main__":
    case_backward(True)