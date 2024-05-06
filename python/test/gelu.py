import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_tpu
torch.manual_seed(1000)

import time
def case1():
    """
    确认torch是异步的
    """
    device = "tpu:0"
    batch = 10
    sequence = 1024
    hidden_size = 768
    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu)
    torch_tpu.tpu.OpTimer_reset()

    t1 = time.time()
    inp_tpu = inp_tpu.to(device, non_blocking=True)
    #inp_tpu = inp_tpu.to(torch.float16, non_blocking=True)
    net_cpu = nn.GELU()

    #out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu)
    # for i in range(100000):
    #     a += i
    out_cpu1 = out_tpu.to("cpu", non_blocking = True)
    #diff = abs(out_cpu - out_tpu.to("cpu", non_blocking = True))
    t2 = time.time()
    out_cpu2 = out_tpu.to("cpu", non_blocking = False)
    t3 = time.time()
    print("t2 - t1 = ", t2 - t1)
    print("t3 - t2 = ", t3 - t2)

    # print("cpu_out")
    # print(inp_cpu.grad.flatten()[:10])
    # print("tpu_out")
    # print(inp_tpu_grad.flatten()[:10])

    print(torch.max(abs(diff)))
    import pdb;pdb.set_trace();

def case_time2():
    """
    确认队列是排在scalar
    """

def case2():
    device = "tpu"
    batch = 8
    sequence = 1024
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp_cpu.to(device)#.half()

    net_cpu = nn.GELU()
    net_tpu = copy.deepcopy(net_cpu).to(device)#.half()

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu).float().to("cpu")

    out_diff = out_cpu - out_tpu
    print("cpu_out")
    print(out_cpu.flatten()[:10])
    print("tpu_out")
    print(out_tpu.flatten()[:10])
    
    print (torch.max(abs(out_diff)))

def test(use_half = True, test_backward = True):
    device = "tpu"
    batch = 1
    sequence = 1024
    hidden_size = 3072

    inp_cpu = torch.randn(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device)
    grad_cpu = torch.randn(batch, sequence, hidden_size)
    grad_tpu = grad_cpu.to(device)

    gelu_cpu = nn.GELU()
    gelu_tpu = copy.deepcopy(gelu_cpu)
    gelu_tpu = gelu_tpu.to(device)

    if use_half:
        inp_tpu = inp_tpu.half()
        grad_tpu = grad_tpu.half()
        gelu_tpu = gelu_tpu.half()

    inp_cpu.requires_grad = True
    inp_tpu.requires_grad = True

    out_cpu = gelu_cpu(inp_cpu)
    out_tpu = gelu_tpu(inp_tpu)

    out_cmp = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_cmp
    print("cpu forward out")
    print(out_cpu.flatten()[:50])
    print("tpu forward out")
    print(out_cmp.flatten()[:50])
    print (torch.max(abs(out_diff)))

    if test_backward:
        out_cpu.backward(grad_cpu)
        out_tpu.backward(grad_tpu)

        inp_tpu_grad = inp_tpu.grad.float().to("cpu")
        grad_diff = inp_cpu.grad - inp_tpu_grad
        print("cpu backward out")
        print(inp_cpu.grad.flatten()[:50])
        print("tpu backward out")
        print(inp_tpu_grad.flatten()[:50])
        print (torch.max(abs(grad_diff)))

if __name__ == "__main__":
    #test(True, True)
    case2()