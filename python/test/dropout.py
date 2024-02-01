import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compare_model_grad, Optimer
import copy

PLUGIN_PATH = "../../build/torch_tpu/libtorch_tpu.so"
torch.ops.load_library(PLUGIN_PATH)
optimer = Optimer(PLUGIN_PATH)

if __name__ == "__main__":
    ###############configure##########
    Batch = 1
    nHead = 1
    Hidden = 8
    sequence = 8
    p_drop = 0.5
    device = "tpu"
    ###################################
    inp = torch.randn((Batch, nHead, sequence, sequence))
    inp_tpu = inp.to(device)
    inp.requires_grad = True
    inp_tpu.requires_grad = True

    ref = torch.ones((Batch, nHead, sequence, sequence)) * 2
    ref_tpu = ref.to(device)

    net = nn.Dropout(p_drop)
    net_tpu = copy.deepcopy(net).to(device)

    print("=====forward====")
    o = net(inp)
    o_t = net_tpu(inp_tpu)
    print("====backward===")
    o.backward(ref)
    o_t.backward(ref_tpu)
    print("====cat result===")
    print("out", o_t.cpu())
    print("inp", inp_tpu.cpu())
    print("in.grad", inp_tpu.grad.cpu())
    print("grad diff",max(inp_tpu.grad.cpu()-inp.grad.cpu()))
