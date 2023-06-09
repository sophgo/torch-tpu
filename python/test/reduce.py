import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)

def case1(use_f16=False):
    """
    reduce sum
    """
    ############## config ###################
    device = "privateuseone"
    h = 999
    w = 111111
    #########################################
    inp = torch.rand(h, w)
    inp_tpu = inp.to(device)
    if use_f16: inp_tpu.half()

    o = torch.sum(inp, dim=0, keepdim=True)
    o_tpu = torch.sum(inp_tpu, dim =0, keepdim=True)

    diff = abs(o - o_tpu.cpu()) #/abs(o_c)
    print("max_diff: ", torch.max(diff))
    print(o[0,:10])
    print(o_tpu.cpu()[0,:10])

if __name__ == "__main__":
    case1(True)