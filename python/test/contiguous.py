import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import Optimer
torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
optimer = Optimer("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = "privateuseone"

def case1():
    a = torch.rand(10,10)
    a_tpu = a.to(device)
    optimer.reset()
    b_tpu = a_tpu.contiguous()
    optimer.dump()

def case_mergeSplitpermute_sequence_256(use_fp16 = False):
    device = "privateuseone"
    batch = 32
    head_num = 12
    hidden_size = 768
    sequence = 256

    inp = torch.randn(batch, sequence, hidden_size * 3)
    inp_tpu = inp.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    print("=======cpu=====")
    q,k,v = inp.split(hidden_size, -1)
    q1 =q.view(batch, sequence, head_num, hidden_size // head_num)
    q2 = q1.permute(0, 2, 1, 3)
    q3 = q2.transpose(-1,-2)
    q4 = q3.contiguous()
    print(q3.stride())
    print(q4.stride())

    print("=======tpu=====")
    q_tpu,_,__ = inp_tpu.split(hidden_size, -1)
    q1_tpu =q_tpu.view(batch, sequence, head_num, hidden_size // head_num)
    q2_tpu = q1_tpu.permute(0, 2, 1, 3)
    q3_tpu = q2_tpu.transpose(-1,-2)
    q4_tpu = q3_tpu.contiguous()
    print(q3_tpu.stride())
    print(q4_tpu.stride())

    print("=====compare======")
    diff = q4 - q4_tpu.cpu()
    print("diff", torch.max(abs(diff)))

if __name__ == "__main__":
    case1()
    #case_mergeSplitpermute_sequence_256(True)
