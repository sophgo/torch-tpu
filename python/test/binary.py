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


if __name__ == "__main__":
    case1()