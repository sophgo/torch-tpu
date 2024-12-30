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
        silu_shape = (x.shape[0], w1.shape[0])
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
        
        # w0x = torch.matmul(x, w0.transpose(-1, -2).contiguous()) #
        # w1x = torch.matmul(x,w1.transpose(-1, -2).contiguous()) #
        
        # silu = F.silu(w1x) # 
        # sigmoid = F.sigmoid(w1x) #
        grad_silu_t = (sigmoid + silu * (1-sigmoid))
        
        grad_tmp = torch.matmul(grad_output, w2_t)  
        grad_w1x = w0x * grad_tmp * grad_silu_t
        grad_w0x = grad_tmp * silu
        grad_w2 = torch.matmul((w0x * silu).transpose(-1,-2).contiguous(), grad_output)
        grad_w1 = torch.matmul(x_t, grad_w1x)
        grad_w0 = torch.matmul(x_t, grad_w0x)
        grad_input = torch.matmul(grad_w0x, w0) + torch.matmul(grad_w1x, w1)

        return grad_input, grad_w0.transpose(-1,-2).contiguous(), grad_w1.transpose(-1,-2).contiguous(), grad_w2

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

class Qwen2MlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, linear_fc1_weight, linear_fc2_weight):
        # hidden_states: [s, b, h]
        # linear_fc1: [2i, h] -> gate_proj [i, h], up_proj [i, h]
        # linear_fc2: [h, i] -> down_proj [h, i]
        gate_proj, up_proj = torch.chunk(linear_fc1_weight, chunks=2, dim=0)
        down_proj = linear_fc2_weight
        down_proj_t = down_proj.t().contiguous()

        s, b, h = hidden_states.shape
        i = down_proj.shape[1]
        calc_shape_silu = (s * b, i)
        calc_shape_output = (s * b, h)

        silu = torch.empty(calc_shape_silu, dtype = hidden_states.dtype, device = hidden_states.device)
        sigmoid = torch.empty(calc_shape_silu, dtype = hidden_states.dtype, device = hidden_states.device)
        w0x = torch.empty(calc_shape_silu, dtype = hidden_states.dtype, device = hidden_states.device)

        output = torch.empty(calc_shape_output, dtype = hidden_states.dtype, device = hidden_states.device)
        x = hidden_states.view(calc_shape_output)

        torch.ops.my_ops.llama_mlp_forward(x,
                                           up_proj,
                                           gate_proj,
                                           down_proj_t,
                                           silu,
                                           sigmoid,
                                           w0x,
                                           output,
                                           True)
        output = output.view(s, b, h)
        ctx.save_for_backward(x, up_proj, gate_proj, down_proj, silu, sigmoid, w0x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, up_proj, gate_proj, down_proj, silu, sigmoid, w0x = ctx.saved_tensors
        y = grad_output.view(x.shape) # [s，b, h] -> [s * b, h]

        grad_silu_t = (sigmoid + silu * (1 - sigmoid)) # [s * b, i]
        grad_tmp = torch.matmul(y, down_proj) # [s * b, h] @ [h, i] = [s * b, i]
        grad_w1x = w0x * grad_tmp * grad_silu_t # [s * b, i]
        grad_w0x = grad_tmp * silu # [s * b, i]

        # x_t = x.t() # [s * b, h] -> [h, s * b]
        # grad_w2 = torch.matmul((w0x * silu).t(), y) # [i, s * b] @ [s * b, h] = [i, h]
        # grad_w1 = torch.matmul(x_t, grad_w1x) # [h, s * b] @ [s * b, i] = [h, i]
        # grad_w0 = torch.matmul(x_t, grad_w0x) # [h, s * b] @ [s * b, i] = [h, i]

        ## a @ b = c <=> b.T @ a.T = c.T
        ## Use this property to avoid .t().contiguous() operation
        ## .t() in matmul is supported so we do not need to .contiguous() after .t()
        grad_w2 = torch.matmul(y.t(), w0x * silu) # [h, s * b] @ [s * b, i] = [h, i]
        grad_w1 = torch.matmul(grad_w1x.t(), x) # [i, s * b] @ [s * b, h] = [i, h]
        grad_w0 = torch.matmul(grad_w0x.t(), x) # [i, s * b] @ [s * b, h] = [i, h]

        grad_input = torch.matmul(grad_w0x, up_proj) + torch.matmul(grad_w1x, gate_proj) # [s * b, i] @ [i, h] = [s * b, h]
        grad_input = grad_input.view(grad_output.shape) # [s * b, h] -> [s, b, h]

        # grad_linear_fc1 = torch.cat([grad_w1, grad_w0], dim=1).t().contiguous()
        # grad_linear_fc2 = grad_w2.t().contiguous()

        ## the previous two .t().contiguous() operations are avoided
        grad_linear_fc1 = torch.cat([grad_w1, grad_w0], dim=0) # [i, h], [i, h] -> [2i, h]
        grad_linear_fc2 = grad_w2

        return grad_input, grad_linear_fc1, grad_linear_fc2

def fuse_llama_mlp():
    import transformers
    def llama_mlp_forward(self, x):
        return LLamaMlpFunc.apply(x, self.up_proj.weight.t().contiguous(), self.gate_proj.weight.t().contiguous(), self.down_proj.weight.t().contiguous())
    transformers.models.llama.modeling_llama.LlamaMLP.forward = llama_mlp_forward

def fuse_qwen2_mlp():
    import megatron
    from megatron.core.parallel_state import get_tensor_model_parallel_group
    def hook(grad):
        if get_tensor_model_parallel_group():
            torch.distributed.all_reduce(grad, group=get_tensor_model_parallel_group())
        return grad
    def qwen2_tpu_mlp_forward(self, hidden_states):
        if self.config.bias_activation_fusion and self.activation_func == F.silu and self.config.gated_linear_unit:
            # note: megatron qwen2 hidden_states shape=(seq_len, batch, hidden_size), different from transformers llama_mlp (batch, seq_len, hidden_size)
            output = Qwen2MlpFunc.apply(hidden_states, self.linear_fc1.weight, self.linear_fc2.weight)
            if get_tensor_model_parallel_group():
                torch.distributed.all_reduce(output, group=get_tensor_model_parallel_group())
                hidden_states.register_hook(hook)
            return output, None
        else:
            ValueError("MegatronDeepSpeedTPU only support silu activation func")
    import megatron_patch
    from megatron_patch.model.qwen2.transformer.mlp import MLP
    megatron_patch.model.qwen2.transformer.mlp.MLP.forward = qwen2_tpu_mlp_forward