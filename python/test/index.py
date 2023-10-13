import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    B = 10
    S = 34
    H = 34
    inp = torch.randn((B,S,H))
    B_ind = torch.LongTensor([1, 2])
    S_ind = torch.randint(0, S, [1,34,4])
    inp_tpu = copy.deepcopy(inp).to(device)
    B_ind_tpu = copy.deepcopy(B_ind).to(device)
    S_ind_tpu = copy.deepcopy(S_ind).to(device)
    
    o_cpu = inp[B_ind]
    o_tpu = inp_tpu[B_ind_tpu]

    diff = o_cpu - o_tpu.cpu()
    print("input : ", inp)
    print("cpu : ", o_cpu)
    print("tpu : ", o_tpu.cpu())
    print(f"max diff : {torch.max(abs(diff))}")
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    case1()