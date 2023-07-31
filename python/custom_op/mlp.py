import os
import torch
import torch.nn as nn

torch.ops.load_library("../../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

class MlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, b1, b2):
        B, M, N = x.shape
        D1 = w1.shape[1]
        D2 = w2.shape[1]
        assert w1.shape == (N, D1)
        assert w2.shape == (D1, D2)
        if b1 != None:
            assert b1.shape == (D1,)
        if b2 != None:
            assert b2.shape == (D2,)
        out1 = torch.empty((B, M, D1), dtype = x.dtype, device = x.device)
        p = torch.empty((B, M, D1), dtype = x.dtype, device = x.device)
        out2 = torch.empty((B, M, D2), dtype = x.dtype, device = x.device)

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
        grad_x = torch.empty(x.shape, dtype = x, device = grad_output.device)
        grad_w1 = torch.empty(w1.shape, dtype = x.dtype, device = grad_output.device)
        grad_w2 = torch.empty(w2.shape, dtype = x.dtype, device = grad_output.device)

        grad_b1 = torch.empty((D1,), dtype = x.dtype, device = grad_output.device)
        grad_b2 = torch.empty((D2,), dtype = x.dtype, device = grad_output.device)

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
    def __init__(self, embed_dim, intermediate_size, has_bias1 = False, has_bias2 = False):
        super().__init__()
        w1 = torch.empty(embed_dim, intermediate_size)
        nn.init.normal_(w1, std=0.02)
        self.w1 = nn.Parameter(w1)
        if has_bias1:
            self.b1 = nn.Parameter(torch.zeros(intermediate_size)) 
        
        w2 = torch.empty(intermediate_size, embed_dim)
        nn.init.normal_(w2, std=0.02)
        self.w2 = nn.Parameter(w2)
        if has_bias2:
            self.b2 = nn.Parameter(torch.zeros(embed_dim)) 

    def forward(self, x):
        return MlpFunc.apply(x, self.w1, self.w2, self.b1, self.b2)
