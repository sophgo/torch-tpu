import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x0, x1):
        y = x0 + x1
        y = y + x0
        y = y + 3.0
        y = y * 4.0
        y = 2.0 - y
        return y

def case1(use_fp16 = False):

    device = "privateuseone"
    shape = [6, 1024, 1, 768]
    shape1 = copy.deepcopy(shape)
    shape1[0] = 1

    inp_cpu = torch.rand(shape)
    inp1_cpu = torch.rand(shape1)
    inp_tpu = copy.deepcopy(inp_cpu).to(device) if not use_fp16 else copy.deepcopy(inp_cpu).to(device).half()
    inp1_tpu = copy.deepcopy(inp1_cpu).to(device) if not use_fp16 else copy.deepcopy(inp1_cpu).to(device).half()

    net_cpu = Model()
    net_tpu = copy.deepcopy(net_cpu)
    net_tpu = net_tpu.to(device)

    out_cpu = net_cpu(inp_cpu, inp1_cpu)
    out_tpu = net_tpu(inp_tpu, inp1_tpu)

    out_tpu = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu
    print("cpu_out")
    print(out_cpu)
    print("tpu_out")
    print(out_tpu)

    print (torch.max(abs(out_diff)))

def const_cases():
    device = "privateuseone"
    tensor0_cpu = torch.tensor([1, 2, 3]).int()
    tensor0_tpu = tensor0_cpu.to(device)
    const = 2
    ans0_cpu = tensor0_cpu + const
    ans1_cpu = tensor0_cpu - const
    ans2_cpu = tensor0_cpu * const
    ans3_cpu = tensor0_cpu / const
    ans0_tpu = tensor0_tpu + const
    ans1_tpu = tensor0_tpu - const
    ans2_tpu = tensor0_tpu * const
    ans3_tpu = tensor0_tpu / const
    print(torch.max(torch.abs(torch.stack([ans0_cpu - ans0_tpu.cpu(),
                                           ans1_cpu - ans1_tpu.cpu(),
                                           ans2_cpu - ans2_tpu.cpu(),
                                           ans3_cpu - ans3_tpu.cpu()]))))

def muldiv_cases(use_fp16 = False):
    device = "privateuseone"
    tensor0_cpu = torch.rand(48, 32).float()
    tensor0_tpu = tensor0_cpu.to(device)
    tensor1_cpu = torch.rand(48, 32).float()
    tensor1_tpu = tensor1_cpu.to(device)
    tensor2_cpu = torch.rand(1, 32).float()
    tensor2_tpu = tensor2_cpu.to(device)
    if use_fp16:
        tensor0_tpu = tensor0_tpu.half()
        tensor1_tpu = tensor1_tpu.half()
    
    ans0_cpu = tensor0_cpu * tensor1_cpu / tensor2_cpu
    ans0_tpu = tensor0_tpu * tensor1_tpu / tensor2_tpu
    
    ans1_cpu = tensor0_cpu / tensor1_cpu * tensor2_cpu
    ans1_tpu = tensor0_tpu / tensor1_tpu * tensor2_tpu

    print(ans0_cpu, ans0_tpu.cpu(), sep="\n")
    print(ans1_cpu, ans1_tpu.cpu(), sep="\n")
    print(torch.max(torch.abs(torch.stack([ans0_cpu - ans0_tpu.cpu(),
                                           ans1_cpu - ans1_tpu.cpu()]))))

def corner_cases():
    device = "privateuseone"
    tensor0_cpu = torch.tensor([1, 2, 3]).float()
    tensor0_tpu = tensor0_cpu.to(device)
    tensor1_cpu = torch.tensor(2).float()
    tensor1_tpu = tensor1_cpu.to(device)
    tensor2 = torch.tensor([2]).float()
    
    ans0_cpu = tensor0_cpu - tensor1_cpu
    ans0_tpu = tensor0_tpu - tensor1_tpu
    
    ans1_cpu = tensor0_cpu - tensor2
    ans1_tpu = tensor0_tpu - tensor2
    print(torch.max(torch.abs(torch.stack([ans0_cpu - ans0_tpu.cpu(),
                                           ans1_cpu - ans1_tpu.cpu()]))))

def dtype_change_cases():
    device = "privateuseone"
    tensor0_cpu = torch.tensor([1, 2, 3]).int()
    tensor0_tpu = tensor0_cpu.to(device)
    tensor1_cpu = torch.tensor(2).float()
    tensor1_tpu = tensor1_cpu.to(device)
    tensor2 = torch.tensor([2]).float()
    
    ans0_cpu = tensor0_cpu - tensor1_cpu
    ans0_tpu = tensor0_tpu - tensor1_tpu
    
    ans1_cpu = tensor0_cpu - tensor2
    ans1_tpu = tensor0_tpu - tensor2
    print(torch.max(torch.abs(ans0_cpu - ans0_tpu.cpu(), 
                              ans1_cpu - ans1_tpu.cpu())))

if __name__ == "__main__":
    case1()
    const_cases()
    muldiv_cases()
    # corner_cases() # not supported now
    # dtype_change_cases() # not supported now