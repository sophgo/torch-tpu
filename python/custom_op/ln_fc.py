import os
import torch
import torch.nn as nn

torch.ops.load_library("../../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class lnMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, gamma, beta):
        D = w.shape[1]
        if x.dim() == 3:
            B, M, N = x.shape
            
            mean = torch.empty((B, M), dtype = x.dtype, device = x.device)
            rstd = torch.empty((B, M), dtype = x.dtype, device = x.device)
            out = torch.empty((B, M, D), dtype = x.dtype, device = x.device)
        else:
            M, N = x.shape
            
            mean = torch.empty((M,), dtype = x.dtype, device = x.device)
            rstd = torch.empty((M,), dtype = x.dtype, device = x.device)
            out = torch.empty((M, D), dtype = x.dtype, device = x.device)
        assert w.shape == (N, D)
        assert gamma.shape == (N,)
        assert beta.shape == (N,)
        if b != None:
            assert b.shape == (D,)

        torch.ops.my_ops.ln_mm_forward(x,
                                     w,
                                     b,
                                     gamma,
                                     beta,
                                     1e-5, 
                                     mean,
                                     rstd,
                                     out)

        ctx.save_for_backward(x, w, mean, rstd)

        return mean, rstd, out 


class lnMatmulBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        w = torch.empty(in_dim, out_dim)
        nn.init.normal_(w, std=0.02)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(out_dim))

        gamma = torch.ones(in_dim,)
        self.gamma = nn.Parameter(gamma)
        self.beta = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return lnMatmulFunc.apply(x, self.w, self.b, self.gamma, self.beta)