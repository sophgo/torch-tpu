import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    H = 50257
    W = 768
    inp = torch.ones((H, W))
    inp_tpu = inp.to(device)#.half()

    o_2  = torch.norm(inp, 2.0)
    ot_2 = torch.norm(inp_tpu, 2.0)
    print("p-2 cpu: ", o_2)
    print("p-2 tpu: ", ot_2.cpu())

    o_1 = torch.norm(inp, 1.0)
    ot_1 = torch.norm(inp_tpu, 1.0)
    print("p-1 cpu: ", o_1)
    print("p-1 tpu: ", ot_1.cpu())


    o_inf = torch.norm(inp, torch.inf)
    ot_inf = torch.norm(inp_tpu, torch.inf)
    print("p-inf cpu: ", o_inf)
    print("p-inf tpu: ", ot_inf.cpu())


if __name__ == "__main__":
    case1()