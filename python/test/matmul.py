import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)

class MyMatmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        return torch.matmul(q, k.transpose(-1,-2))
        # return torch.matmul(q, k)
    

def check_matmul():
    device = "privateuseone"
    use_half = False

    b = 2
    m = 12
    l = 16
    d = 64

    q_cpu = torch.rand(b, m, l, d)
    k_cpu = torch.rand(b, m, l, d)
    # k_cpu = torch.rand(b, m, d, l)
    # q_cpu = torch.rand(b, l, d)
    # k_cpu = torch.rand(b, l, d)

    q_tpu = q_cpu.to(device)
    k_tpu = k_cpu.to(device)

    ref = torch.ones(b, m, l, l)
    # ref = torch.rand(b, l, l)
    ref_tpu = ref.to(device)

    net = MyMatmul()
    net_tpu = copy.deepcopy(net)
    net_tpu.to(device)

    if use_half:
        q_tpu = q_tpu.half()
        k_tpu = k_tpu.half()
        net_tpu = net_tpu.half()

    q_cpu.requires_grad = True
    k_cpu.requires_grad = True
    q_tpu.requires_grad = True
    k_tpu.requires_grad = True

    out_cpu = net(q_cpu, k_cpu)
    out_tpu = net_tpu(q_tpu, k_tpu)

    print(torch.max(abs(out_cpu - out_tpu.cpu())))

    out_cpu.backward(ref)
    out_tpu.backward(ref_tpu)

    import pdb;pdb.set_trace()

    print(torch.max(abs(q_cpu.grad - q_tpu.grad.cpu())))
    print(torch.max(abs(k_cpu.grad - k_tpu.grad.cpu())))

    # assert ref @ k_cpu == q_cpu.grad
    # assert ref_tpu.cpu() @ k_tpu.cpu() == q_tpu.grad.cpu()
    # assert (q_cpu.transpose(-1,-2) @ ref).transpose(-1,-2) == k_cpu.grad
    # assert (q_tpu.transpose(-1,-2).cpu() @ ref_tpu.cpu()).transpose(-1,-2) == k_tpu.grad.cpu()



if __name__ == "__main__":
    check_matmul()

    
    