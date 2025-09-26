
import torch
import torch.nn.functional as F
import megatron
from megatron import core
from megatron.core.transformer.mlp import MLP


class MegatronQwen2MlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, linear_fc1_weight, linear_fc2_weight,buf_fwd, buf_bwd):
        # hidden_states: [s, b, h]
        # linear_fc1: [2i, h] -> gate_proj [i, h], up_proj [i, h]
        # linear_fc2: [h, i] -> down_proj [h, i]

        gate_proj, up_proj = torch.chunk(linear_fc1_weight, chunks=2, dim=0)
        down_proj = linear_fc2_weight
        down_proj_t = down_proj.t().contiguous()

        #assign the buffer for w1x, w0x and output
        # w1x: [s * b, i], w0x: [s * b, i], output: [s * b, h]
        s, b, h = hidden_states.shape
        i = down_proj.shape[1]
        calc_shape_silu = (s * b, i)
        calc_shape_output = (s * b, h)
        total_elems = s * b * (2 * i + h)
        buf = buf_fwd[:total_elems]
        sizes = [s * b * i, s * b * i, s * b * h]
        chunks = buf.split(sizes)

        w1x, w0x, output = (
          chunks[0].view(calc_shape_silu),
          chunks[1].view(calc_shape_silu),
          chunks[2].view(calc_shape_output),
        )

        x = hidden_states.view(calc_shape_output)

        torch.ops.my_ops.llama_mlp_forward(x,
                                           up_proj,
                                           gate_proj,
                                           down_proj_t,
                                           None,
                                           None,
                                           None,
                                           w1x,
                                           w0x,
                                           output,
                                           True)
        output = output.view(s, b, h)
        ctx.save_for_backward(x, up_proj, gate_proj, down_proj, w0x, w1x, buf_bwd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, up_proj, gate_proj, down_proj, w0x, w1x, buf_bwd = ctx.saved_tensors

        y = grad_output.view(x.shape) # [s，b, h] -> [s * b, h]
        use_torch = False
        if(use_torch):
            sigmoid = torch.sigmoid(w1x)
            silu = w1x * sigmoid

            grad_silu_t = (sigmoid + silu * (1 - sigmoid)) # [s * b, i]
            grad_tmp = torch.matmul(y, down_proj) # [s * b, h] @ [h, i] = [s * b, i]
            grad_w1x =  w0x * grad_tmp * grad_silu_t  # [s * b, i]
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

        else:
            if x.dim() == 3:
              x = x.view(-1, x.size(2)).contiguous()
            w0x_shape = (w0x.shape[-2], w0x.shape[-1])

            #assign the buffer for grad_input, grad_w0, grad_w1, grad_w2 and grad_tmp
            # grad_input: [s * b, h], grad_w0: [i, h], grad_w1: [i, h], grad_w2: [h, i], grad_tmp: [s * b, i]
            buf_elems = x.numel() + up_proj.numel() + gate_proj.numel() + down_proj.numel() + w0x.shape[-2]*w0x.shape[-1]
            buf_tot = buf_bwd[:buf_elems]
            sizes = [x.numel(), up_proj.numel(), gate_proj.numel(), down_proj.numel(), w0x.shape[-2]*w0x.shape[-1]]
            chunks = buf_tot.split(sizes)
            grad_input, grad_w0, grad_w1, grad_w2, grad_tmp = (
                chunks[0].view_as(x),
                chunks[1].view_as(up_proj),
                chunks[2].view_as(gate_proj),
                chunks[3].view_as(down_proj),
                chunks[4].view(w0x_shape),)

            torch.ops.my_ops.mlp_backward(x,
                                            up_proj,
                                            gate_proj,
                                            down_proj,
                                            w0x,
                                            w1x,
                                            y,
                                            grad_tmp,
                                            grad_input,
                                            grad_w0,
                                            grad_w1,
                                            grad_w2)
            grad_input = grad_input.view(grad_output.shape)
            grad_linear_fc1 = torch.cat([grad_w1, grad_w0], dim=0)
            grad_linear_fc2 = grad_w2

        return grad_input, grad_linear_fc1, grad_linear_fc2, None, None

def fuse_megatron_mlp():
    from megatron.core.parallel_state import get_tensor_model_parallel_group
    def hook(grad):
        if get_tensor_model_parallel_group():
            torch.distributed.all_reduce(grad, group=get_tensor_model_parallel_group())
        return grad
    def qwen2_tpu_mlp_forward(self, hidden_states):
        if self.config.bias_activation_fusion and self.activation_func == F.silu and self.config.gated_linear_unit:
            # note: megatron qwen2 hidden_states shape=(seq_len, batch, hidden_size), different from transformers llama_mlp (batch, seq_len, hidden_size)
            b, s, h = hidden_states.shape #s for seq_len, b for batch, h for hidden_size
            i = self.linear_fc2.weight.shape[1]  # hidden_size // 2
            max_fwd = s*b * (2 * i + h) #for three forward buffers
            max_bwd = s*b * (h + i) + 3 * i * h #for five backward buffers
            #if the bufffer is less than the max_fwd or max_bwd, we need to resize it
            if self.mlp_scratch_fwd.numel() < max_fwd:
                self.mlp_scratch_fwd = torch.empty(
                    max_fwd, dtype=self.linear_fc1.weight.dtype, device=self.linear_fc1.weight.device
                )

            if self.mlp_scratch_bwd.numel() < max_bwd:
                self.mlp_scratch_bwd = torch.empty(
                    max_bwd, dtype=self.linear_fc1.weight.dtype, device=self.linear_fc1.weight.device
                )
            buf_fwd = self.mlp_scratch_fwd
            buf_bwd = self.mlp_scratch_bwd
            output = MegatronQwen2MlpFunc.apply(
                hidden_states,
                self.linear_fc1.weight,
                self.linear_fc2.weight,
                buf_fwd,
                buf_bwd
            )
            if get_tensor_model_parallel_group():
                torch.distributed.all_reduce(output, group=get_tensor_model_parallel_group())
            if hidden_states.requires_grad:
                hidden_states.register_hook(hook)
            return output, None
        else:
            ValueError("MegatronDeepSpeedTPU only support silu activation func")
    _orig_mlp_init = MLP.__init__
    def _mlp_init_with_buf(self, *args, **kwargs):
        _orig_mlp_init(self, *args, **kwargs)
        device = self.linear_fc1.weight.device
        dtype = self.linear_fc1.weight.dtype
        #preset a minimum buffer for forward and backward
        self.register_buffer('mlp_scratch_fwd',
           torch.empty(1, dtype=dtype, device=device),
            persistent=False
        )
        self.register_buffer('mlp_scratch_bwd',
            torch.empty(1, dtype=dtype, device=device),
            persistent=False
        )
    MLP.__init__ = _mlp_init_with_buf
    megatron.core.transformer.mlp.MLP.forward = qwen2_tpu_mlp_forward
