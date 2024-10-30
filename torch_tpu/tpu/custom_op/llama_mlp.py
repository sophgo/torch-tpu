import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models

class LLamaMlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2):
        x_ori = x
        #ctx.save_for_backward(x, w0, w1, w2)
        x_shape = x.shape
        if x.dim() == 3:
           x = x.view(-1, x.size(2)).contiguous()
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)
        silu_shape = (x.shape[0], w1.shape[1])
        silu = torch.empty(silu_shape, dtype = x.dtype, device = x.device)
        sigmoid = torch.empty(silu_shape, dtype = x.dtype, device = x.device)
        m0 = torch.empty(silu_shape, dtype = x.dtype, device = x.device)
        
        torch.ops.my_ops.llama_mlp_forward(x,
                                     w0,
                                     w1,
                                     w2,
                                     silu,
                                     sigmoid,
                                     m0,
                                     output,
                                     True)
        if output.shape != x_shape:
           output = output.view(x_shape).contiguous()
        ctx.save_for_backward(x_ori, w0, w1, w2, silu, sigmoid, m0)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w0, w1, w2, silu, sigmoid, w0x = ctx.saved_tensors
        silu_shape = (x.shape[0], x.shape[1], silu.shape[-1])
        silu = silu.view(silu_shape).contiguous()
        sigmoid = sigmoid.view(silu_shape).contiguous()
        w0x = w0x.view(silu_shape).contiguous()
        
        x_t =x.transpose(-1, -2).contiguous()
        w2_t = w2.t().contiguous()
        # w0x = torch.matmul(x, w0) #
        # w1x = torch.matmul(x,w1) #
        
        # silu = F.silu(w1x) # 
        # sigmoid = F.sigmoid(w1x) #
        grad_silu_t = (sigmoid + silu * (1-sigmoid))
        
        grad_tmp = torch.matmul(grad_output, w2_t)  
        grad_w1x = w0x * grad_tmp * grad_silu_t
        grad_w0x = grad_tmp * silu
        grad_w2 = torch.matmul((w0x * silu).transpose(-1,-2).contiguous(), grad_output)
        grad_w1 = torch.matmul(x_t, grad_w1x)
        grad_w0 = torch.matmul(x_t, grad_w0x)
        grad_input = torch.matmul(grad_w0x, w0.t().contiguous()) + torch.matmul(grad_w1x, w1.t().contiguous())

        return grad_input, grad_w0, grad_w1, grad_w2

class LLamaMlpBlock(nn.Module):
    def __init__(self, embed_dim, intermediate_size):
        super().__init__()
        w0 = torch.empty(embed_dim, intermediate_size)
        w1 = torch.empty(embed_dim, intermediate_size)
        w2 = torch.empty(intermediate_size, embed_dim)
        nn.init.normal_(w0, std=0.02)
        nn.init.normal_(w1, std=0.02)
        nn.init.normal_(w2, std=0.02)
        self.w0 = nn.Parameter(w0)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, x):
        return LLamaMlpFunc.apply(x, self.w0, self.w1, self.w2)

def fuse_llama_mlp():
    import transformers
    def llama_mlp_forward(self, x):
        return LLamaMlpFunc.apply(x, self.up_proj.weight.t().contiguous(), self.gate_proj.weight.t().contiguous(), self.down_proj.weight.t().contiguous())
    transformers.models.llama.modeling_llama.LlamaMLP.forward = llama_mlp_forward
