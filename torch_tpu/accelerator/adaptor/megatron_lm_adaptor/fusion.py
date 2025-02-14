# apply torch_tpu fused kernels
import torch_tpu
from torch_tpu.tpu.custom_op.adam import fuse_torch_adam
from torch_tpu.tpu.custom_op.rmsnorm import fuse_megatron_qwen2_rmsnorm
from torch_tpu.tpu.custom_op.llama_mlp import fuse_megatron_qwen2_mlp
from torch_tpu.tpu.custom_op.llama_attn_qkv import fuse_megatron_qwen2_attn_qkv

fuse_torch_adam()
fuse_megatron_qwen2_rmsnorm()
fuse_megatron_qwen2_mlp()
fuse_megatron_qwen2_attn_qkv()

# disable unsupported cuda fused kernels
import megatron
from megatron import core
megatron.core.jit.jit_fuser = lambda func, *args, **kwargs: func

from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = lambda *args, **kwargs: False

from megatron import training
megatron.training.training.set_jit_fusion_options = lambda: None

from megatron import legacy
megatron.legacy.fused_kernels.load = lambda args: None