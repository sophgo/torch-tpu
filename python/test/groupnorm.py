from tkinter import N
import torch
import torch.nn as nn
import copy
from utils import compare_model_grad, Optimer
torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
OPT = Optimer()
if __name__ == "__main__":
    device = "privateuseone"
    N = 1
    C = 512
    H = 64
    W = 64
    NUM_GROUPS = 32
    inp = torch.randn(N,C,H,W)
    inp_tpu = inp.to(device).half()
    
    net = nn.GroupNorm(num_channels=C, num_groups=NUM_GROUPS, eps=1e-6)
    net_tpu = copy.deepcopy(net).to(device).half()
    res_cpu = net(inp)
    OPT.reset()
    res_tpu = net_tpu(inp_tpu)
    OPT.dump()
    # print("cpu ======")
    # print(res_cpu)
    # print("tpu ======")
    # print(res_tpu.cpu())

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)
    print("cpu:", res_cpu.flatten()[idx])
    print("tpu:", res_tpu.cpu().flatten()[idx])
