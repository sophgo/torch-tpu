import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

class RMSNorm(nn.Module):
    def __init__(self, d=0, axis=-1., eps=1e-8, with_scale=False, with_bias=False):
        super(RMSNorm, self).__init__()
        self.d = d
        self.axis = axis
        self.eps = eps
        self.with_bias = with_bias
        self.with_scale = with_scale
        if self.with_scale:
            self.scale = nn.Parameter(torch.rand(self.d))
        if self.with_bias:
            self.bias = nn.Parameter(torch.rand(self.d))

    def forward(self, x):
        ms = torch.mean(torch.square(x), dim=self.axis, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        y = x / rms
        if self.with_scale:
            y *= self.scale
        if self.with_bias:
            y += self.bias
        return y

class RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias, axis, eps):
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)

        torch.ops.my_ops.rmsnorm_forward(x,
                                     scale,
                                     bias,
                                     output,
                                     axis,
                                     eps)
        return output

class RMSNormBlock(nn.Module):
    def __init__(self, scale=None, bias=None, axis=-1, eps=1e-8):
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        return RMSNormFunc.apply(x, self.scale, self.bias, self.axis, self.eps)

def check_rmsnorm():
    batch = 256
    hidden_size = 1536
    axis = 3
    eps = 1e-5

    net_cpu = RMSNorm(d=hidden_size, axis=axis, eps=eps, with_bias=True, with_scale=True)

    x = torch.randn((batch, 1, 1, hidden_size), requires_grad=False)
    x_tpu = x.half().to(device)

    out_cpu = net_cpu(x)

    scale = None
    bias = None
    if 'scale' in net_cpu.state_dict():
        scale = net_cpu.state_dict()['scale'].clone().detach().contiguous().requires_grad_(False).half().to(device)
    if 'bias' in net_cpu.state_dict():
        bias = net_cpu.state_dict()['bias'].clone().detach().contiguous().requires_grad_(False).half().to(device)

    net_tpu = RMSNormBlock(axis=axis, eps=eps, scale=scale, bias=bias)
    out_tpu = net_tpu(x_tpu)
    out_tpu = out_tpu.to("cpu").float()

    out_diff = out_cpu - out_tpu
    print(out_cpu[0][0][0][:50])
    print(out_tpu[0][0][0][:50])
    print (torch.max(abs(out_diff)))
    return

if __name__=="__main__":
    check_rmsnorm()
