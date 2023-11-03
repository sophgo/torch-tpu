import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy


torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

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
    batch = 16
    hidden_size = 8192
    axis = 3
    eps = 1e-5

    net_cpu = RMSNorm(axis=axis, eps=eps)
    net_tpu = RMSNormBlock(axis=axis, eps=eps)

    x = torch.randn((batch, 1, 1, hidden_size), requires_grad=False)
    x_tpu = x.to(device).half()

    out_cpu = net_cpu(x)
    out_tpu = net_tpu(x_tpu)
    out_tpu = out_tpu.float().to("cpu")

    out_diff = out_cpu - out_tpu
    print(out_cpu[0][0][0][:50])
    print(out_tpu[0][0][0][:50])
    print (torch.max(abs(out_diff)))
    return

if __name__=="__main__":
    check_rmsnorm()
