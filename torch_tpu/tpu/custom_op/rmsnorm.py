import torch
import torch.nn as nn
import os

# now rmsnorm with CMODEL_FAST_EXEC=1 has problem
class cmodel_slow_exec:
    def __init__(self, wait_tensor):
        self.prev = os.environ.get("CMODEL_FAST_EXEC", None)
        self.wait_tensor = wait_tensor
        
    def __enter__(self):
        if not self.prev is None:
            self.wait_tensor.cpu()#wait for async:
            os.environ["CMODEL_FAST_EXEC"] = "0"
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.prev is not None:
            if os.environ.get("ENABLE_PMU", "0") != "1":
                self.wait_tensor.cpu() # wait for sync. only affect cmodel
            os.environ["CMODEL_FAST_EXEC"] = self.prev

class RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias, axis, eps):
        ctx.save_for_backward(x, scale, bias)
        ctx.axis = axis
        ctx.eps = eps
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)

        with cmodel_slow_exec(output):
            torch.ops.my_ops.rmsnorm_forward(x,
                                        scale,
                                        bias,
                                        output,
                                        axis,
                                        eps)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(grad_output) if x.requires_grad else None
        grad_scale = torch.zeros_like(scale) if scale is not None and scale.requires_grad else None
        grad_bias = torch.zeros_like(bias) if bias is not None and bias.requires_grad else None
        rms = torch.zeros_like(grad_output)
        
        with cmodel_slow_exec(rms):
            torch.ops.my_ops.rmsnorm_backward(
                grad_output,
                x,
                scale,
                bias,
                rms,
                grad_input,
                grad_scale,
                grad_bias,
                ctx.axis,
                ctx.eps)
        return grad_input, grad_scale, grad_bias, None, None

class RMSNormBlock(nn.Module):
    def __init__(self, hidden_size, axis=-1, eps=1e-8):
        super().__init__()
        scale = torch.empty(hidden_size)
        bias = torch.empty(hidden_size)
        nn.init.normal_(scale, std=0.02)
        nn.init.normal_(bias, std=0.02)
        self.scale = nn.Parameter(scale)
        self.bias = nn.Parameter(bias)
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        return RMSNormFunc.apply(x, self.scale, self.bias, self.axis, self.eps)

def llama_rmsnorm_forward(self, hidden_states):
    dim = hidden_states.dim() - 1
    return RMSNormFunc.apply(hidden_states, self.weight, None, dim, self.variance_epsilon)

def fuse_llama_rmsnorm():
    import transformers
    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = llama_rmsnorm_forward
 