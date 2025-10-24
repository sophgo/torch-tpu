import torch
import torch_tpu
import os
import sys
import copy

import transformers.modeling_attn_mask_utils
from top_utest import TensorComparator
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
from transformers import Qwen2ForCausalLM, Qwen2Config
os.environ["CMODEL_FAST_EXEC"]="1"

from torch_tpu.tpu.custom_op.llama_attn_qkv import fuse_qwen2_attn_qkv
from torch_tpu.tpu.custom_op.llama_mlp import fuse_qwen2_mlp
from torch_tpu.tpu.custom_op.rmsnorm import fuse_qwen2_rmsnorm

#replace transformers func because lacking tril op
from typing import Optional
@staticmethod
def make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window - 1

        i = torch.arange(tgt_len, dtype=torch.int32, device=device).unsqueeze(1)
        j = torch.arange(tgt_len, dtype=torch.int32, device=device).unsqueeze(0)
        condition = j < torch.minimum(i + diagonal + 1, torch.tensor(tgt_len, device=device))
        mask.masked_fill_(condition, torch.finfo(dtype).min)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
import transformers
transformers.modeling_attn_mask_utils.AttentionMaskConverter._make_causal_mask = staticmethod(make_causal_mask)

qwen2_config = {
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 131072,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "transformers_version": "4.37.2",
    "vocab_size": 152064
}

def case1():
    seed = 2260
    torch.manual_seed(seed)
    device = "tpu:0"

    config = Qwen2Config(**qwen2_config)
    qwen_model = Qwen2ForCausalLM(config)
    qwen_model_tpu = copy.deepcopy(qwen_model).half().to(device)

    batch = 1
    status = False
    for seq_len in [128]:
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        input_ids_tpu = copy.deepcopy(input_ids).to(device)

        output_cpu = qwen_model(input_ids)

        fuse_qwen2_rmsnorm()
        fuse_qwen2_mlp()
        fuse_qwen2_attn_qkv()
        output_tpu = qwen_model_tpu(input_ids_tpu)

        comparator = TensorComparator()
        status = comparator.cmp_result(output_cpu["logits"].detach().float(), output_tpu["logits"].cpu().detach().float())

    if status:
        print(f"Qwen2 is correct")
    else:
        print(f"Qwen2 is wrong")
        sys.exit(255)

    return status

if __name__ == "__main__":
    case1()