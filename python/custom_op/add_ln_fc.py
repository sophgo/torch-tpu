import os
import torch
import torch.nn as nn

torch.ops.load_library("../../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class AddlnMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, w, b, gamma, beta):
        D = w.shape[1]

        if x1.dim() == 3:
            B, M, N = x1.shape
            mean = torch.empty((B, M), dtype = x1.dtype, device = x1.device)
            rstd = torch.empty((B, M), dtype = x1.dtype, device = x1.device)
            out = torch.empty((B, M, D), dtype = x1.dtype, device = x1.device)
        else:
            M, N = x1.shape
            mean = torch.empty((M,), dtype = x1.dtype, device = x1.device)
            rstd = torch.empty((M,), dtype = x1.dtype, device = x1.device)
            out = torch.empty((M, D), dtype = x1.dtype, device = x1.device)
        out_add = torch.empty(x1.shape, dtype = x1.dtype, device = x1.device)

        assert x1.shape == x2.shape
        assert w.shape == (N, D)
        assert gamma.shape == (N,)
        assert beta.shape == (N,)
        if b != None:
            assert b.shape == (D,)

        torch.ops.my_ops.add_ln_mm_forward(x1,
                                     x2,
                                     w,
                                     b,
                                     gamma,
                                     beta,
                                     1e-5,
                                     out_add,
                                     mean,
                                     rstd,
                                     out)

        ctx.save_for_backward(x1, x2, w, mean, rstd)

        return out_add, mean, rstd, out 


class AddlnMatmulBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        w = torch.empty(in_dim, out_dim)
        nn.init.normal_(w, std=0.02)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(out_dim))

        gamma = torch.ones(in_dim,)
        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(torch.zeros(in_dim))



    def forward(self, x1, x2):
        return AddlnMatmulFunc.apply(x1, x2, self.w, self.b, self.gamma, self.beta)