import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tpu
import sys
import os

device = "tpu:0"

def switch_ppl_flag(flag):
    return "0" if flag == "1" else "1"

def case1():
    class MixNet(nn.Module):

        def forward(self, no_use=None):
            A = torch.ones(1024, 1024, dtype=torch.float16).to(device)
            B = torch.ones(1024, 1024, dtype=torch.float16).to(device)

            os.environ["USE_PPL"] = switch_ppl_flag("0")
            B = B * 2.0
            B = B * 20.0
            os.environ["USE_PPL"] = switch_ppl_flag(os.environ["USE_PPL"])
            C = A + B
            C = C * 2.0
            os.environ["USE_PPL"] = switch_ppl_flag(os.environ["USE_PPL"])
            D = C - 5.0
            D = D - 3.0
            os.environ["USE_PPL"] = switch_ppl_flag(os.environ["USE_PPL"])
            E = D - 3.0
            E = E - 3.0
            return E

    class MixNetCpu(nn.Module):
        def forward(self, no_use=None):
            A = torch.ones(1024, 1024, dtype=torch.float16)
            B = torch.ones(1024, 1024, dtype=torch.float16)
            B = B * 2.0
            B = B * 20.0
            C = A + B
            C = C * 2.0
            D = C - 5.0
            D = D - 3.0
            E = D - 3.0
            E = E - 3.0
            return E
    tpu_net = MixNet()
    cpu_net = MixNetCpu()
    tpu_res = tpu_net(None)
    cpu_res = cpu_net(None)
    assert torch.allclose(tpu_res.cpu(), cpu_res)

if __name__ == "__main__":
    case1()