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

        ctx.save_for_backward(out_add, w, mean, rstd, gamma, beta)

        return out_add, out
    
    @staticmethod
    def backward(ctx, grad_add, grad_output):
        out_add, w, mean, rstd, gamma, beta = ctx.saved_tensors

        D = w.shape[1]
        if out_add.dim() == 3:
            B, M, N = out_add.shape
            out_ln_cpu = ((out_add.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).unsqueeze(1).cpu() + beta.unsqueeze(0).unsqueeze(1).cpu()
            grad_out_ln = torch.matmul(grad_output, w.unsqueeze(0).transpose(-1,-2))
        else:
            M, N = out_add.shape
            out_ln_cpu = ((out_add.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).cpu() + beta.unsqueeze(0).cpu()
            grad_out_ln = torch.matmul(grad_output, w.transpose(-1,-2))

        out_ln = out_ln_cpu.to(device)

        grad_out_add = torch.ones(out_add.shape, dtype = out_add.dtype, device = grad_output.device)
        grad_gamma = torch.ones((N,), dtype = out_add.dtype, device = grad_output.device)
        grad_beta = torch.ones((N,), dtype = out_add.dtype, device = grad_output.device)

        grad_w = torch.matmul(out_ln.transpose(-1,-2), grad_output)
        grad_b = grad_output.reshape(-1, D).sum(0)
        
        torch.ops.my_ops.add_ln_mm_backward(grad_out_ln,
                                        out_add,
                                        mean.unsqueeze(-1),
                                        rstd.unsqueeze(-1),
                                        gamma,
                                        grad_out_add,
                                        grad_gamma,
                                        grad_beta)
        
        return grad_out_add, grad_out_add, grad_w, grad_b, grad_gamma, grad_beta


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