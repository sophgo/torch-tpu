import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1(use_fp16):
    shape = (1, 4, 64, 64)
    loss = nn.MSELoss(reduction='mean')
    input = torch.ones(shape, requires_grad=True)
    target = torch.ones(shape) * 2
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

def case_backward(use_fp16 =False):
    shape = (1, 4, 64, 64)
    loss = nn.MSELoss(reduction='mean')
    input = torch.rand(shape, requires_grad=True)
    target = torch.rand(shape)
    if use_fp16:
        input = input.half()
        target = target.half()
        loss = loss.half()
    inp_tpu = input.clone().detach().to(device)
    tar_tpu = target.to(device)

    input.requires_grad = True
    inp_tpu.requires_grad = True

    output = loss(input, target) * 2
    out_tpu = loss(inp_tpu, tar_tpu) * 2
    print("diff: ", torch.sum(output - out_tpu.cpu()))
    print("out: ", output)
    print("out_tpu: ",out_tpu.cpu())

    print("==backward")
    output.backward()
    out_tpu.backward()


    inp_tpu_grad = inp_tpu.grad.to("cpu")
    diff = torch.max(abs(input.grad - inp_tpu_grad))
    print("cpu_grad")
    print(input.grad.flatten()[:10])
    print("tpu_grad")
    print(inp_tpu_grad.flatten()[:10])
    print(torch.max(abs(diff)))

if __name__ == "__main__":
    #case1(use_fp16=True)
    case_backward()