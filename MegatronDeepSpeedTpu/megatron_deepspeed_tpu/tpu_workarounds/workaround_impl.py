from .workaround_helper import *
import math
import os

## avoid 64bit tensors
torch.nn.Module.__call__ = wrap_if_arguments(has_64bit_tensor, wrap_64bit, torch.nn.Module.__call__)

torch.Tensor.expand = wrap_if_arguments(has_64bit_tensor, wrap_64bit, torch.Tensor.expand) # wrap_64bit(torch.Tensor.expand)
torch.norm = wrap_if_arguments(has_64bit_tensor, wrap_64bit, torch.norm) # wrap_64bit(torch.norm)
torch.Tensor.norm = wrap_if_arguments(has_64bit_tensor, wrap_64bit, torch.Tensor.norm) # wrap_64bit(torch.Tensor.norm)

# destroy 64bit dtype convert
torch.Tensor.long = wrap_64bit(torch.Tensor.long)
torch.Tensor.double = wrap_64bit(torch.Tensor.double)

## avoid unsupported ops
# using cpu
torch.cumsum = wrap_cpu(torch.cumsum) # aten::cumsum.out not implemented

torch.arange = wrap_cpu(torch.arange) # sg2260 not implemented
# torch.nonzero = wrap_cpu(torch.nonzero) # sg2260 not implemented
# torch.Tensor.nonzero = wrap_cpu(torch.Tensor.nonzero)
if_nonzero = lambda x: isinstance(x[0][1], torch.Tensor) and x[0][1].dtype == torch.bool # will dispatch to nonzero and has problem
torch.Tensor.__getitem__ = wrap_if_arguments(if_nonzero, wrap_cpu, torch.Tensor.__getitem__)

torch.nn.functional.cross_entropy = wrap_cpu(torch.nn.functional.cross_entropy) # sg2260 not implemented

torch.index_select = wrap_cpu(torch.index_select) # wrong answer

if_broadcast = lambda x: isinstance(x[0][1], torch.Tensor) and x[0][0].shape != x[0][1].shape # bcbinary wrong answer
torch.add = wrap_if_arguments(if_broadcast, wrap_cpu, torch.add)
torch.sub = wrap_if_arguments(if_broadcast, wrap_cpu, torch.sub)
torch.mul = wrap_if_arguments(if_broadcast, wrap_cpu, torch.mul)
torch.div = wrap_if_arguments(if_broadcast, wrap_cpu, torch.div)
torch.Tensor.add = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.add)
torch.Tensor.sub = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.sub)
torch.Tensor.mul = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.mul)
torch.Tensor.div = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.div)
torch.Tensor.add_ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.add_)
torch.Tensor.sub_ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.sub_)
torch.Tensor.mul_ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.mul_)
torch.Tensor.div_ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.div_)
torch.Tensor.__add__ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.__add__)
torch.Tensor.__sub__ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.__sub__)
torch.Tensor.__mul__ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.__mul__)
torch.Tensor.__truediv__ = wrap_if_arguments(if_broadcast, wrap_cpu, torch.Tensor.__truediv__)

torch.Tensor.__getitem__ = wrap_contiguous(torch.Tensor.__getitem__)

# torch.isinf = wrap_cpu(torch.isinf) # wrong answer
# torch.Tensor.isinf = wrap_cpu(torch.Tensor.isinf)
# torch.isnan = wrap_cpu(torch.isnan) # wrong answer
# torch.Tensor.isnan = wrap_cpu(torch.Tensor.isnan)

# torch.logical_or = wrap_cpu(torch.logical_or) # assert when shape large
# torch.Tensor.logical_or = wrap_cpu(torch.Tensor.logical_or)

n_args_is_one = lambda x: sum(map(len, x)) == 1
torch.max = wrap_if_arguments(n_args_is_one, wrap_cpu, torch.max, wrap_fp32) # aten::max not implemented; fp16 not supported
torch.Tensor.max = wrap_if_arguments(n_args_is_one, wrap_cpu,torch.Tensor.max, wrap_fp32)
torch.min = wrap_if_arguments(n_args_is_one, wrap_cpu, torch.min, wrap_fp32) # aten::min not implemented; fp16 not supported
torch.Tensor.min = wrap_if_arguments(n_args_is_one, wrap_cpu, torch.Tensor.min, wrap_fp32)

# has_unused_core = lambda x: any(map(lambda dim: math.ceil(dim / math.ceil(dim / 8)) < 8, (x[0][0].shape[-2], x[0][1].shape[-2], x[0][1].shape[-1])))
# torch.matmul = wrap_if_arguments(has_unused_core, wrap_cpu, torch.matmul) # fixed # matmul multi core has problem when not every core is used

# torch.index_add = wrap_cpu(torch.index_add) # aten::index_add.out not implemented (used in custom embedding_backward

torch.Tensor.copy_ = wrap_non_blocking(torch.Tensor.copy_) # non_blocking not supported
torch.Tensor.to = wrap_non_blocking(torch.Tensor.to)

# tanh fp16 not supported
# aten::tanh_backward.grad_input not implemented
class tanh_fw_fp32_bw_cpu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = wrap_fp32(torch.ops.aten.tanh)(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return wrap_cpu(torch.ops.aten.tanh_backward)(grad_output, output)

torch.tanh = tanh_fw_fp32_bw_cpu.apply
torch.Tensor.tanh = lambda x: tanh_fw_fp32_bw_cpu.apply(x)

# embedding forward has problem in certain shape
# embedding backward sg2260 not implemented
class embedding_fw_tpu_bw_cpu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, input, padding_idx, scale_grad_by_freq, sparse):
        ctx.save_for_backward(input)
        ctx.num_weights = weight.shape[0]
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx.sparse = sparse
        return wrap_cpu(torch.ops.aten.embedding)(weight, input, padding_idx, scale_grad_by_freq, sparse)

    @staticmethod
    def backward(ctx, grad_output):
        input,= ctx.saved_tensors
        return wrap_cpu(torch.ops.aten.embedding_backward)(grad_output.reshape(-1, grad_output.shape[-1]),
                                                        input.reshape(-1),
                                                        ctx.num_weights,
                                                        ctx.padding_idx,
                                                        ctx.scale_grad_by_freq,
                                                        ctx.sparse), None, None, None, None

torch.embedding = embedding_fw_tpu_bw_cpu.apply

# layernorm backward return 0 when shape small, nan when fp16 shape large
class layernorm_fw_tpu_bw_cpu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, cudnn_enable=False):
        ctx.normalized_shape = normalized_shape
        ctx.output_mask = [input is not None, weight is not None, bias is not None]
        output, mean, rstd = torch.ops.aten.native_layer_norm(input, normalized_shape, weight, bias, eps)
        ctx.save_for_backward(input, weight, bias, mean, rstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mean, rstd = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = wrap_cpu(torch.ops.aten.native_layer_norm_backward)(grad_output, input, ctx.normalized_shape, mean, rstd, weight, bias, ctx.output_mask)
        return grad_input, None, grad_weight, grad_bias, None, None

torch.layer_norm = layernorm_fw_tpu_bw_cpu.apply

# expand backward when dim = 0. Tired of defining all backward functions, changed in tpu-train