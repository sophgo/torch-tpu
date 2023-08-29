import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1(use_fp16):
    shape = (1, 4, 64, 64)
    loss = nn.MSELoss(reduction='mean')
    input = torch.ones(shape, requires_grad=True)
    target = torch.rand(shape)
    if use_fp16:
        input = input.half()
        target = target.half()
        loss = loss.half()
    inp_tpu = input.clone().detach().to(device)
    tar_tpu = target.to(device)

    #input.requires_grad = True
    #inp_tpu.requires_grad = True

    output = loss(input, target)
    # output.backward()
    print(output)

    # import pdb
    # pdb.set_trace()
    out_tpu = loss(inp_tpu, tar_tpu)
    # out_tpu.backward()
    print(out_tpu.cpu())

    print(torch.sum(output - out_tpu.cpu()))

    # inp_tpu_grad = inp_tpu.grad.to("cpu")
    # diff = torch.max(abs(input.grad - inp_tpu_grad))
    # print("cpu_out")
    # print(inp_tpu.grad.flatten()[:10])
    # print("tpu_out")
    # print(inp_tpu_grad.flatten()[:10])

    # print(torch.max(abs(diff)))

if __name__ == "__main__":
    case1(use_fp16=True)