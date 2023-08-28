import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    B = 1 
    S = 77
    H = 768
    inp = torch.randn((B,S,H))
    B_ind = torch.LongTensor([0])
    S_ind = torch.randint(0, S, [1])
    import pdb;pdb.set_trace()
    inp_tpu = copy.deepcopy(inp).to(device)
    B_ind_tpu = copy.deepcopy(B_ind).to(device)
    S_ind_tpu = copy.deepcopy(S_ind).to(device)
    
    o_cpu = inp[B_ind, S_ind,]
    o_tpu = inp_tpu[B_ind_tpu, S_ind_tpu,]

    diff = o_cpu - o_tpu.cpu()
    print(diff)


if __name__ == "__main__":
    case1()