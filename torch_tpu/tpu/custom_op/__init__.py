__all__ = [
     "AddlnMatmulBlock", "AttentionBlock", "LLamaMlpBlock",
     "lnMatmulBlock", "MlpBlock", "RMSNormBlock"
]

from .add_ln_fc import AddlnMatmulBlock
from .attention import AttentionBlock
from .llama_mlp import LLamaMlpBlock
from .ln_fc import lnMatmulBlock
from .mlp import MlpBlock
from .rmsnorm import RMSNormBlock

import torch
import torch_tpu

def register_tpu_ops_api():
     torch_tpu.tpu_format_cast          = torch.ops.tpu.tpu_format_cast
     
     torch_tpu.tpu_add_ln_mm_forward    = torch.ops.my_ops.add_ln_mm_forward
     torch_tpu.tpu_add_ln_mm_backward   = torch.ops.my_ops.add_ln_mm_backward
     
     torch_tpu.tpu_attn_forward         = torch.ops.my_ops.attn_forward
     torch_tpu.tpu_attn_backward        = torch.ops.my_ops.attn_backward
     
     torch_tpu.tpu_llama_mlp_forward    = torch.ops.my_ops.llama_mlp_forward
     
     torch_tpu.tpu_ln_mm_forward   = torch.ops.my_ops.ln_mm_forward
     torch_tpu.tpu_ln_mm_backward  = torch.ops.my_ops.ln_mm_backward

     torch_tpu.tpu_mlp_forward     = torch.ops.my_ops.mlp_forward 
     torch_tpu.tpu_mlp_backward    = torch.ops.my_ops.mlp_backward

     torch_tpu.tpu_rmsnorm_forward = torch.ops.my_ops.rmsnorm_forward
register_tpu_ops_api()