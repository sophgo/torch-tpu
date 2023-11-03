import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compare_model_grad, Optimer

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"
OPT = Optimer()

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

    OPT.reset()
    Y_t, mean_t, std_t = torch.native_group_norm(inp_tpu, weight, bias, N, C, H*W, group_num, eps);
    OPT.dump()

    # print("p-2 cpu: ", Y, mean, std)
    # print("p-2 tpu: ", Y_t.cpu(), mean_t.cpu(), std_t.cpu())

    diff = abs(Y - Y_t.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)
    print("cpu:", Y.flatten()[idx])
    print("tpu:", Y_t.cpu().flatten()[idx])


if __name__ == "__main__":
    case1()