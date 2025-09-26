import os
import torch
import torch_tpu
import torch.nn as nn
import torch.nn.functional as F
import copy
import transformers.models
import pdb

class LLamaMlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2, use_cpu_fw=False, return_mid_tensor=False, use_cpu_bw=False):
        x_ori = x
        #ctx.save_for_backward(x, w0, w1, w2)
        x_shape = x.shape
        if x.dim() == 3:
           x = x.view(-1, x.size(2)).contiguous()
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)
        silu_shape = (x.shape[0], w1.shape[0])
        silu = torch.empty(silu_shape, dtype = x.dtype, device = x.device)
        sigmoid = torch.empty(silu_shape, dtype = x.dtype, device = x.device)
        fc1 = torch.empty(silu_shape, dtype = x.dtype, device = x.device)
        m0 = torch.empty(silu_shape, dtype = x.dtype, device = x.device)

        if use_cpu_fw:
            w1_trans = w1.transpose(0, 1)
            fc1 = torch.matmul(x, w1_trans)
            sigmoid = torch.sigmoid(fc1)
            silu = fc1 * sigmoid
            w0_trans = w0.transpose(0, 1)
            fc0 = torch.matmul(x, w0_trans)
            m0 = torch.mul(fc0, silu)
            output = torch.matmul(m0, w2)
        else:
            torch.ops.my_ops.llama_mlp_forward(x,
                                        w0,
                                        w1,
                                        w2,
                                        None,
                                        None,
                                        None,
                                        fc1,
                                        m0,
                                        output,
                                        True)
        if output.shape != x_shape:
           output = output.view(x_shape).contiguous()

        ctx.save_for_backward(x_ori, w0, w1, w2, m0, fc1)

        ctx.use_cpu_bw = use_cpu_bw

        if return_mid_tensor:
            return output, m0, fc1
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w0, w1, w2, w0x, w1x = ctx.saved_tensors
        use_cpu_bw = ctx.use_cpu_bw
        x_shape = x.shape
        w2_t = w2.transpose(-1,-2).contiguous()
        if use_cpu_bw:
          sigmoid = torch.sigmoid(w1x)
          silu = w1x * sigmoid
          w0x = w0x.view(silu.shape).contiguous()


          grad_silu_t = (sigmoid + silu * (1-sigmoid))
          grad_tmp = torch.matmul(grad_output, w2_t)
          grad_w1x = w0x * grad_tmp * grad_silu_t

          grad_w0x = grad_tmp * silu
          grad_w2 = torch.matmul((w0x * silu).transpose(-1,-2).contiguous(), grad_output)
          grad_w1 = torch.matmul(grad_w1x.transpose(-1,-2).contiguous(), x)
          grad_w0 = torch.matmul(grad_w0x.transpose(-1,-2).contiguous(), x)
          grad_input = torch.matmul(grad_w0x, w0) + torch.matmul(grad_w1x, w1)
        else:
          if x.dim() == 3:
            x = x.view(-1, x.size(2)).contiguous()
          if grad_output.shape != x.shape:
            grad_output = grad_output.view(x.shape).contiguous()
          grad_input = torch.empty_like(x)
          grad_w0 = torch.empty(w0.size(), device=x.device, dtype=x.dtype)
          grad_w1 = torch.empty(w1.size(), device=x.device, dtype=x.dtype)
          grad_w2 = torch.empty(w2_t.size(), device=x.device, dtype=x.dtype)

          w0x_shape = (w0x.shape[-2], w0x.shape[-1])
          grad_tmp = torch.full(w0x_shape,0, device=x.device, dtype=x.dtype)
          torch.ops.my_ops.mlp_backward(x,
                                          w0,
                                          w1,
                                          w2_t,
                                          w0x,
                                          w1x,
                                          grad_output,
                                          grad_tmp,
                                          grad_input,
                                          grad_w0,
                                          grad_w1,
                                          grad_w2)
          grad_w2 = grad_w2.transpose(-1,-2).contiguous()

        if(grad_input.shape != x_shape):
          grad_input = grad_input.view(x_shape).contiguous()

        return grad_input, grad_w0, grad_w1, grad_w2, None, None, None

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
        return LLamaMlpFunc.apply(x, self.up_proj.weight, self.gate_proj.weight, self.down_proj.weight.t().contiguous())
    transformers.models.llama.modeling_llama.LlamaMLP.forward = llama_mlp_forward

#transformers qwen2
def fuse_qwen2_mlp():
    import transformers
    def qwen2_mlp_forward(self, x):
        return LLamaMlpFunc.apply(x, self.up_proj.weight, self.gate_proj.weight, self.down_proj.weight.t().contiguous())
    transformers.models.qwen2.modeling_qwen2.Qwen2MLP.forward = qwen2_mlp_forward