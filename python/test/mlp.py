import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy

from utils import Optimer, compare_model_grad, compare_model_weight


torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

class Conv1D(nn.Module):
    """
    Basically works like a linear layer

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
    

class GPT2Mlp(nn.Module):
    def __init__(self, embed_dim, intermediate_size):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = F.gelu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x
    

class MlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, b1, b2):
        B, M, N = x.shape
        D1 = w1.shape[1]
        D2 = w2.shape[1]
        assert w1.shape == (N, D1)
        assert w2.shape == (D1, D2)
        assert b1.shape == (D1,)
        assert b2.shape == (D2,)
        out1 = torch.empty(B, M, D1).type_as(x).to(device)
        p = torch.empty(B, M, D1).type_as(x).to(device)
        out2 = torch.empty(B, M, D2).type_as(x).to(device)

        torch.ops.my_ops.mlp_forward(x,
                                     w1,
                                     w2,
                                     b1,
                                     b2,
                                     out1,
                                     p,
                                     out2)

        ctx.save_for_backward(x, w1, w2, out1, p)

        return out2

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, w2, out1, p = ctx.saved_tensors

        B, M, N = x.shape
        D1 = w1.shape[1]
        D2 = w2.shape[1]
        grad_output.to(device)
        grad_x = torch.ones(x.shape, dtype = x.dtype).to(device)
        grad_w1 = torch.ones(w1.shape, dtype = x.dtype).to(device)
        grad_w2 = torch.ones(w2.shape, dtype = x.dtype).to(device)

        grad_b1 = torch.ones((D1,), dtype = x.dtype).to(device)
        grad_b2 = torch.ones((D2,), dtype = x.dtype).to(device)

        torch.ops.my_ops.mlp_backward(grad_output,
                                        x,
                                        w1,
                                        w2,
                                        out1,
                                        p,
                                        grad_x,
                                        grad_w1,
                                        grad_w2,
                                        grad_b1,
                                        grad_b2)

        return grad_x, grad_w1, grad_w2, grad_b1, grad_b2


class MlpBlock(nn.Module):
    def __init__(self, w1, w2, b1, b2):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def forward(self, x):
        return MlpFunc.apply(x, self.w1, self.w2, self.b1, self.b2)


def check_mlp():
    batch_size = 4
    length = 10
    embed_dim = 32
    intermediate_size = 128
    net_cpu = GPT2Mlp(embed_dim, intermediate_size)

    # w1 = torch.tensor(copy.deepcopy(net_cpu.state_dict()['c_fc.weight']), requires_grad=True).to(device)
    w1 = net_cpu.state_dict()['c_fc.weight'].clone().detach().transpose(0,1).requires_grad_(True).to(device)   # TODO
    b1 = net_cpu.state_dict()['c_fc.bias'].clone().detach().requires_grad_(True).to(device)
    w2 = net_cpu.state_dict()['c_proj.weight'].clone().detach().transpose(0,1).requires_grad_(True).to(device)
    b2 = net_cpu.state_dict()['c_proj.bias'].clone().detach().requires_grad_(True).to(device)

    net_tpu = MlpBlock(w1, w2, b1, b2)

    print("=====forward======")
    x = torch.randn(batch_size, length, embed_dim, requires_grad=True).to(device)
    out_tpu = net_tpu(x)
    out_cpu = net_cpu(x.cpu())
    out_diff = out_cpu - out_tpu.float().to("cpu")
    print (torch.max(abs(out_diff)))

    print("=====backward======")
    ref_tpu = torch.ones(batch_size, length, b2.shape[0]).to(device)
    ref_cpu = ref_tpu.cpu()
    out_tpu.backward(ref_tpu)
    out_cpu.backward(ref_cpu)

    compare_model_grad(net_cpu, net_tpu)
    compare_model_weight(net_cpu, net_tpu)

check_mlp()

