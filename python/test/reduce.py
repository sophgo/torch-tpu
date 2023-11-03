import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case_2d_sum(use_f16=False):
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

def case_3d_sum(use_f16=False):
    ############## config ###################
    device = "privateuseone"
    batch = 32
    sequence = 256
    hidden_size = 768
    #########################################
    inp = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp.to(device)
    if use_f16: inp_tpu.half()

    o = torch.sum(inp, dim=0, keepdim=False)
    o_tpu = torch.sum(inp_tpu, dim =0, keepdim=False)

    diff = abs(o - o_tpu.cpu()) #/abs(o_c)
    print("max_diff: ", torch.max(diff))
    print(o[0,:10])
    print(o_tpu.cpu()[0,:10])

def case_FP16_override():
    inp = torch.Tensor([30000] * 10 + [-30000] * 10).to(device).half()
    print(inp.cpu())
    o1 = torch.sum(inp)
    print(o1.float().cpu())


if __name__ == "__main__":
    case_3d_sum(True)