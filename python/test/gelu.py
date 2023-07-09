import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

def case1():
    """
    gelu backward 
    """
    device = "privateuseone"
    batch = 8
    sequence = 1024
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device)#.half()
    
    #inp_cpu.retain_grad = True
    inp_cpu.requires_grad = True
    #inp_tpu.retain_grad = True
    inp_tpu.requires_grad = True

    grad_cpu = torch.rand(batch, sequence, hidden_size)
    grad_tpu = grad_cpu.to(device).half()

    net_cpu = nn.GELU()
    net_tpu = copy.deepcopy(net_cpu).to(device)#.half()

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu)

    out_cpu.backward(grad_cpu)
    out_tpu.backward(grad_tpu)

    inp_tpu_grad = inp_tpu.grad.to("cpu")
    diff = torch.max(abs(inp_cpu.grad - inp_tpu_grad))
    print("cpu_out")
    print(inp_cpu.grad.flatten()[:10])
    print("tpu_out")
    print(inp_tpu_grad.flatten()[:10])

    print(torch.max(abs(diff)))
    
def case2():
    device = "privateuseone"
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

if __name__ == "__main__":
    # case2()
    case1()