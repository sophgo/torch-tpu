import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_tpu
torch.manual_seed(1000)

def case1(use_fp16=False):
    """
    conv3d forward
    """
    device = "tpu"

    IC = 3
    OC = 1280
    KS = [2, 14, 14]
    BS = 1

    inp_cpu = torch.rand((BS, 3, 2, 14, 14), dtype=torch.float32)
    inp_tpu = copy.deepcopy(inp_cpu)
    if use_fp16:
        inp_tpu = inp_tpu.to(torch.float16)
    diff = inp_cpu - inp_tpu.cpu()
    print("debugjw in  diff", torch.max(abs(diff)))
    inp_tpu = inp_tpu.to(device)

    net_cpu = nn.Conv3d(in_channels=IC, out_channels=OC, kernel_size=KS, stride=KS, bias=False)
    net_tpu = copy.deepcopy(net_cpu)
    if use_fp16:
        net_tpu.weight = nn.Parameter(net_cpu.weight.to(torch.float16), requires_grad=False)
    net_tpu = net_tpu.to(device)

    out_cpu = net_cpu(inp_cpu)
    out_tpu = net_tpu(inp_tpu)
    print(out_cpu.shape, out_tpu.shape)
    diff = out_cpu - out_tpu.cpu()
    print(diff)
    print("debugjw out diff", torch.max(abs(diff)))


if __name__ == "__main__":
    case1(use_fp16 = True)
