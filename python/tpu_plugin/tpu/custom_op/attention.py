import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class AttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_attn, w_proj, b_attn, b_proj, bias, masked_bias, num_head):
        B, M, N = x.shape
        H = num_head
        D_attn = w_attn.shape[1]
        assert w_attn.shape == (N, D_attn)
        assert w_proj.shape == (D_attn//3, N)  # TODO
        if b_attn != None:
            assert b_attn.shape == (D_attn,)
        if b_proj != None:
            assert b_proj.shape == (D_attn//3,)

        q = torch.empty((B, M, D_attn//3), dtype = x.dtype, device = x.device)
        k = torch.empty((B, M, D_attn//3), dtype = x.dtype, device = x.device)
        v = torch.empty((B, M, D_attn//3), dtype = x.dtype, device = x.device)

        softmax_out = torch.empty((B, H, M, M), dtype = x.dtype, device = x.device)
        soft_v = torch.empty((B, H, M, D_attn//(3*H)), dtype = x.dtype, device = x.device)
        out = torch.empty((B, M, N), dtype = x.dtype, device = x.device)

        torch.ops.my_ops.attn_forward(x,
                                     w_attn,
                                     w_proj,
                                     b_attn,
                                     b_proj,
                                     q,
                                     k,
                                     v,
                                     softmax_out,
                                     soft_v,
                                     out)

        ctx.save_for_backward(x, w_attn, w_proj, q, k, v, softmax_out, soft_v, bias)

        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w_attn, w_proj, q, k, v, softmax_out, soft_v, bias = ctx.saved_tensors
        B, M, N = x.shape
        H = softmax_out.shape[1]
        D_attn = w_attn.shape[1]

        grad_x = torch.ones(x.shape, dtype=x.dtype, device = grad_output.device)
        grad_w_attn = torch.ones(w_attn.shape, dtype=x.dtype, device = grad_output.device)
        grad_w_proj = torch.ones(w_proj.shape, dtype=x.dtype, device = grad_output.device)

        grad_b_attn = grad_b_proj = None
        if ctx.needs_input_grad[3]:
            grad_b_attn = torch.ones((D_attn,), dtype=x.dtype, device = grad_output.device)
        if ctx.needs_input_grad[4]:
            grad_b_proj = torch.ones((D_attn//3,), dtype=x.dtype, device = grad_output.device)

        torch.ops.my_ops.attn_backward(grad_output,
                                        x,
                                        w_attn,
                                        w_proj,
                                        q, 
                                        k, 
                                        v,
                                        softmax_out,
                                        soft_v,
                                        bias,
                                        grad_x,
                                        grad_w_attn,
                                        grad_w_proj,
                                        grad_b_attn,
                                        grad_b_proj)

        return grad_x, grad_w_attn, grad_w_proj, grad_b_attn, grad_b_proj, None, None, None




class AttentionBlock(nn.Module):
    def __init__(self, w_attn, w_proj, b_attn, b_proj, bias, masked_bias, num_head):
        super().__init__()
        self.w_attn = w_attn
        self.w_proj = w_proj
        self.b_attn = b_attn
        self.b_proj = b_proj
        self.bias = bias
        self.masked_bias = masked_bias
        self.num_head = num_head

    def forward(self, x):
        return AttnFunc.apply(x, self.w_attn, self.w_proj, self.b_attn, self.b_proj, self.bias, self.masked_bias, self.num_head)