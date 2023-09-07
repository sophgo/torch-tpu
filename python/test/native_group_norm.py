import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    N = 2
    C = 2
    H = 3
    W = 3
    group_num = 2
    eps = 1e-5
    weight = torch.randn(C)
    bias = torch.randn(C)

    inp = torch.randn(N, C, H, W)
    inp_tpu = inp.to(device)#.half()

    Y, mean, std = torch.native_group_norm(inp, weight, bias, N, C, H*W, group_num, eps);

    Y_t, mean_t, std_t = torch.native_group_norm(inp_tpu, weight, bias, N, C, H*W, group_num, eps);


    print("p-2 cpu: ", Y, mean, std)
    print("p-2 tpu: ", Y_t.cpu(), mean_t.cpu(), std_t.cpu())

    diff = Y - Y_t.cpu()
    print(torch.max(torch.abs(diff)))


if __name__ == "__main__":
    case1()