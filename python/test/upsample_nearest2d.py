import torch
import torch.nn as nn
import copy
from utils import compare_model_grad, Optimer
import torch_tpu
torch.manual_seed(1000)
OPT = Optimer()
if __name__ == "__main__":
    device = "tpu"
    N = 4
    C = 64
    H = 128
    W = 128
    inp = torch.randn(N,C,H,W)
    inp_tpu = inp.to(device)
    inp.requires_grad = True
    inp_tpu.requires_grad = True
    
    net = nn.Upsample(size=None, scale_factor=4, mode='nearest', align_corners=None)
    net_tpu = copy.deepcopy(net).to(device)
    res_cpu = net(inp)
    OPT.reset()
    res_tpu = net_tpu(inp_tpu)
    OPT.dump()
    # print("cpu ======")
    # print(res_cpu)
    # print("tpu ======")
    # print(res_tpu.cpu())

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)
    print("cpu:", res_cpu.flatten()[idx])
    print("tpu:", res_tpu.cpu().flatten()[idx])

    grad_o = torch.rand(res_cpu.shape)
    grad_o_tpu = grad_o.to(device)

    res_cpu.backward(grad_o)
    res_tpu.backward(grad_o_tpu)

    diff = inp.grad - inp_tpu.grad.cpu()
    print(torch.max(abs(diff)))
    # print(torch.argmax(diff))
    print(torch.nn.functional.cosine_similarity(inp.grad.flatten(),inp_tpu.grad.cpu().flatten(),dim=0))
    print(torch.min(torch.nn.functional.cosine_similarity(inp.grad,inp_tpu.grad.cpu(),dim=3)))
    compare_model_grad(net, net_tpu.cpu())