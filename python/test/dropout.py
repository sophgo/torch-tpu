import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compare_model_grad, Optimer
import copy

PLUGIN_PATH = "../../libtorch_plugin/build/liblibtorch_plugin.so"
torch.ops.load_library(PLUGIN_PATH)
optimer = Optimer(PLUGIN_PATH)

if __name__ == "__main__":
    ###############configure##########
    Batch = 8
    nHead = 12
    Hidden = 768
    sequence = 1024
    p_drop = 0.1
    device = "privateuseone"
    ###################################
    inp = torch.randn((Batch, nHead, sequence, sequence))
    inp_tpu = inp.to(device)

    net = nn.Dropout(p_drop)
    net_tpu = copy.deepcopy(net).to(device)

    o = net(inp)
    o_t = net_tpu(inp_tpu)

    diff = o - o_t.cpu()
    print(torch.max(torch.abs(diff)))
