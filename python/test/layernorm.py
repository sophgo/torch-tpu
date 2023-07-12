import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

def test(use_half = False, test_backward = False):
    device = "privateuseone"
    batch = 8
    sequence = 1024
    hidden_size = 768
    layer_norm_epsilon = 1e-5

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp_cpu.to(device)
    grad_cpu = torch.rand(batch, sequence, hidden_size)
    grad_tpu = grad_cpu.to(device)

    inp_cpu.require_grad = True
    inp_tpu.require_grad = True

    ln_cpu = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
    ln_tpu = copy.deepcopy(ln_cpu)
    ln_tpu = ln_tpu.to(device)

    if use_half:
        inp_tpu = inp_tpu.half()
        grad_tpu = grad_tpu.half()
        ln_tpu = ln_tpu.half()

    out_cpu = ln_cpu(inp_cpu)
    out_tpu = ln_tpu(inp_tpu)

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
    test(True, True)
