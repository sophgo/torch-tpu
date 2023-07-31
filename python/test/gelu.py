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
    batch = 1
    sequence = 1024
    hidden_size = 768

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device).to(torch.bfloat16)
    
    #inp_cpu.retain_grad = True
    inp_cpu.requires_grad = True
    #inp_tpu.retain_grad = True
    inp_tpu.requires_grad = True

    grad_cpu = torch.rand(batch, sequence, hidden_size)
    grad_tpu = grad_cpu.to(device).to(torch.bfloat16)

    net_cpu = nn.GELU()
    net_tpu = copy.deepcopy(net_cpu).to(device).to(torch.bfloat16)

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu)

    # import pdb;pdb.set_trace();
    out_cpu.backward(grad_cpu)
    out_tpu.backward(grad_tpu)

    inp_tpu_grad = inp_tpu.grad.to("cpu")
    diff = torch.max(abs(inp_cpu.grad - inp_tpu_grad))
    print("cpu_out")
    print(inp_cpu.grad.flatten()[:10])
    print("tpu_out")
    print(inp_tpu_grad.flatten()[:10])

    print(torch.max(abs(diff)))
    import pdb;pdb.set_trace();
    
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

def test(use_half = False, test_backward = False):
    device = "privateuseone"
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
    case1()