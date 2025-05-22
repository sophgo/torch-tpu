import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from enum import Enum
import random
import numpy as np
import pdb

import torch_tpu
import torch_tpu.tpu
torch.manual_seed(300)
random.seed(300)
device = "tpu:0"

class AttentionMode(Enum):
    CONTINUOUS_PREFILL = 0
    CONTINUOUS_DECODE = 1
    PAGED_PREFILL = 2
    PAGED_DECODE = 3

@dataclass
class MLATensors:
    seqlen: torch.Tensor
    WUQ: torch.Tensor
    WUKV: torch.Tensor
    Q: torch.Tensor
    KV: torch.Tensor
    PE: torch.Tensor
    kv_cache: torch.Tensor
    pe_cache: torch.Tensor
    idx_theta: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    output: torch.Tensor
    mask: torch.Tensor = None
    WUQ_scale: torch.Tensor = None
    WUKV_scale: torch.Tensor = None
    block_tables: torch.Tensor = None
    slots: torch.Tensor = None
    paged_kvcache: torch.Tensor = None
    paged_pecache: torch.Tensor = None

@dataclass
class MLAConfig:
    batch: int
    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    block_size: int
    softmax_scale: float
    dtype: torch.dtype
    weight_dtype: torch.dtype
    context_len: int
    attention_mode: AttentionMode
    max_cache_size: int = 8192
    paged_block_size: int = 16
    mask_size: int = 0

    def block_index_to_slots(self, block_idx, block_size, context_len):
        flat_slots_idx = []
        for b in range(block_idx.size(0)):
            for s in range(block_idx.size(1)):
                block_end = min(block_size, (context_len - block_size * s) if s == block_idx.size(1) - 1 else block_size)
                flat_slots_idx.extend([block_idx[b, s] * block_size + j for j in range(block_end)])
        return flat_slots_idx

    def random_tensors(self):
        assert self.max_cache_size >= self.context_len + 1, "max_cache_size should be larger than context_len + 1"
        seqlen = torch.tensor([self.context_len for _ in range(self.batch)], dtype = torch.int32)
        WUQ = torch.randn((self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), self.q_lora_rank),
                        dtype = self.dtype).to(self.weight_dtype)
        WUKV = torch.randn((self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank),
                        dtype = self.dtype).to(self.weight_dtype)
        seq = max(seqlen).item() if self.attention_mode in (
            AttentionMode.CONTINUOUS_PREFILL,
            AttentionMode.PAGED_PREFILL,
        ) else 1
        Q = torch.randn((self.batch, seq, self.q_lora_rank), dtype = self.dtype)
        KV = torch.randn((self.batch, seq, self.kv_lora_rank), dtype = self.dtype)
        PE = torch.randn((self.batch, seq, self.qk_rope_head_dim), dtype = self.dtype)
        kv_cache = torch.randn((self.batch, self.max_cache_size, self.kv_lora_rank), dtype = self.dtype)
        pe_cache = torch.randn((self.batch, self.max_cache_size, self.qk_rope_head_dim), dtype = self.dtype)
        idx_theta = gen_rope(self.batch * seq, self.qk_rope_head_dim).to(self.dtype)
        cos = torch.cos(idx_theta).contiguous().to(self.dtype)
        sin = torch.sin(idx_theta).contiguous().to(self.dtype)
        output = torch.randn((self.batch, seq, self.num_heads, self.v_head_dim), dtype = self.dtype)

        mask = None
        if self.attention_mode == AttentionMode.CONTINUOUS_PREFILL \
            or self.attention_mode == AttentionMode.PAGED_PREFILL:
            self.mask_size = seq
            neg_inf = -1e4
            mask = torch.full((self.mask_size, self.mask_size), neg_inf, dtype = self.dtype)
            mask = torch.triu(mask, diagonal=1)

        paged_kvcache = None
        paged_pecache = None
        block_tables = None
        slots = None
        if self.attention_mode == AttentionMode.PAGED_DECODE:
            block_per_batch = int(math.ceil((self.context_len + 1) / self.paged_block_size))
            total_block_num = max(block_per_batch * self.batch, 512)
            paged_kvcache = torch.empty((total_block_num, self.paged_block_size, self.kv_lora_rank), dtype = self.dtype)
            paged_pecache = torch.empty((total_block_num, self.paged_block_size, self.qk_rope_head_dim), dtype = self.dtype)
            block_idx = random.sample(range(total_block_num), block_per_batch * self.batch)
            block_tables = torch.tensor(block_idx, dtype = torch.int32).view(self.batch, -1)
            slots_idx = [block_idx[(b+1) * block_per_batch - 1] * self.paged_block_size + \
                         self.context_len % self.paged_block_size for b in range(self.batch)]
            slots = torch.tensor(slots_idx, dtype = torch.int32)
            # fill paged kvcache and pecahce
            flat_slots_idx = self.block_index_to_slots(block_tables, self.paged_block_size, self.context_len)
            indices = torch.tensor(flat_slots_idx, dtype = torch.int64)
            indices = indices.view(self.batch, -1)
            for b in range(self.batch):
                paged_kvcache.view(-1, self.kv_lora_rank)[indices[b]] = kv_cache[b,:len(indices[b])]
                paged_pecache.view(-1, self.qk_rope_head_dim)[indices[b]] = pe_cache[b,:len(indices[b])]
        elif self.attention_mode == AttentionMode.PAGED_PREFILL:
            block_per_batch = int(math.ceil(self.context_len / self.paged_block_size))
            total_block_num = max(block_per_batch * self.batch, 512)
            paged_kvcache = torch.empty((total_block_num, self.paged_block_size, self.kv_lora_rank), dtype = self.dtype)
            paged_pecache = torch.empty((total_block_num, self.paged_block_size, self.qk_rope_head_dim), dtype = self.dtype)
            block_idx = random.sample(range(total_block_num), block_per_batch * self.batch)
            block_tables = torch.tensor(block_idx, dtype = torch.int32).view(self.batch, -1)
            slots = block_tables


        WUQ_scale = None
        WUKV_scale = None
        if self.weight_dtype == torch.float8_e4m3fn:
            WUQ_scale = torch.randn(int(math.ceil(WUQ.shape[0] / self.block_size)),
                                    int(math.ceil(WUQ.shape[1] / self.block_size)),
                                    dtype = self.dtype)
            WUKV_scale = torch.randn(int(math.ceil(WUKV.shape[0] / self.block_size)),
                                    int(math.ceil(WUKV.shape[1] / self.block_size)),
                                    dtype = self.dtype)

        return MLATensors(
            seqlen, WUQ, WUKV, Q, KV, PE, kv_cache, pe_cache, idx_theta, cos, sin, output,
            mask, WUQ_scale, WUKV_scale, block_tables, slots, paged_kvcache, paged_pecache)


def gen_rope(Ntotal, head_size):
    theta = torch.arange(0, head_size, 2).float() / head_size
    theta = 1. / 10000 ** theta
    theta = theta.view(1, -1).squeeze()  # (1, d/2)
    seq_idx = torch.arange(Ntotal, dtype=theta.dtype)
    if Ntotal > 1:
        seq_idx = seq_idx.view(-1, 1).squeeze()  # (seq_len, 1)
    idx_theta = torch.einsum('i,j->ij', seq_idx, theta)  # (seq_len, d/2)
    idx_theta = idx_theta.view(idx_theta.shape[0],1,-1)
    idx_theta = idx_theta.repeat(1,1,2)
    return idx_theta

def rotate_every_two(x):
    x_swapped = torch.cat((-x[:, :, (int)(x.shape[-1]/2):], x[:, :, :(int)(x.shape[-1]/2)]), dim=-1)
    return x_swapped

def apply_rope(x, cos, sin):
    # RoPE in deepseek is interleaved, so we need to swap the elements
    # to get the correct rotation.
    assert x.dim() == 3
    batch, num_heads, head_dim = x.shape
    assert head_dim % 2 == 0
    x = (x.view(batch, num_heads, -1, 2).transpose(2, 3).contiguous()).view(batch, num_heads, -1)
    x = (x * cos) + (rotate_every_two(x) * sin)
    return x


def fp8_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.
    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size for tiling. Defaults to 128.
    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.
    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    m, n = [(i + block_size - 1) // block_size for i in x.size()]
    pad = [0 for i in range(4)]
    if N % block_size:
        pad[1] = block_size - N % block_size
    if M % block_size:
        pad[3] = block_size - M % block_size
    if pad:
        x = F.pad(x, pad)
    converted_x = x.view(m, block_size, n, block_size).to(s.dtype)
    out = (converted_x * s.view(m, 1, n, 1))
    if pad:
        return out.reshape(m * block_size, n * block_size)[:M, :N]
    else:
        return out.reshape(M, N)

class MLA(nn.Module):
    def __init__(self, config: MLAConfig):
        super(MLA, self).__init__()
        self.config = config
    def forward(self):
        RuntimeError("reimplent for MLATpu or MLACpu")

class MLATpu(MLA):
    def __init__(self, config: MLAConfig):
        super(MLATpu, self).__init__(config)

    def forward(self, tensors: MLATensors):
        if (tensors.WUQ.dtype == torch.float8_e4m3fn or tensors.WUQ.dtype == torch.float8_e5m2):
            # 0: normal attention
            if self.config.attention_mode == AttentionMode.CONTINUOUS_DECODE \
                or self.config.attention_mode == AttentionMode.CONTINUOUS_PREFILL:
                torch.ops.my_ops.latent_attention_fp8(
                    tensors.output, tensors.Q, tensors.KV, tensors.PE, tensors.WUQ, tensors.WUKV,
                    tensors.kv_cache, tensors.pe_cache, tensors.cos, tensors.sin, tensors.WUQ_scale,
                    tensors.WUKV_scale, tensors.mask, tensors.seqlen, self.config.num_heads,
                    self.config.q_lora_rank, self.config.kv_lora_rank, self.config.qk_nope_head_dim,
                    self.config.qk_rope_head_dim, self.config.v_head_dim, self.config.mask_size,
                    self.config.block_size, self.config.max_cache_size, self.config.softmax_scale,
                    self.config.attention_mode.value)
            elif self.config.attention_mode == AttentionMode.PAGED_DECODE \
                or self.config.attention_mode == AttentionMode.PAGED_PREFILL:
                # paged attention
                torch.ops.my_ops.paged_latent_attention_fp8(
                    tensors.output, tensors.Q, tensors.KV, tensors.PE, tensors.WUQ, tensors.WUKV,
                    tensors.paged_kvcache, tensors.paged_pecache, tensors.cos, tensors.sin, tensors.WUQ_scale,
                    tensors.WUKV_scale, tensors.block_tables, tensors.slots,
                    tensors.mask, tensors.seqlen, self.config.num_heads,
                    self.config.q_lora_rank, self.config.kv_lora_rank, self.config.qk_nope_head_dim,
                    self.config.qk_rope_head_dim, self.config.v_head_dim, self.config.mask_size,
                    self.config.block_size, tensors.block_tables.shape[1],
                    self.config.paged_block_size, self.config.softmax_scale,
                    self.config.attention_mode.value)
        else:
            if self.config.attention_mode == AttentionMode.CONTINUOUS_DECODE \
                or self.config.attention_mode == AttentionMode.CONTINUOUS_PREFILL:
                torch.ops.my_ops.latent_attention(
                    tensors.output, tensors.Q, tensors.KV, tensors.PE,
                    tensors.WUQ, tensors.WUKV, tensors.kv_cache, tensors.pe_cache,
                    tensors.cos, tensors.sin, tensors.mask, tensors.seqlen, self.config.num_heads,
                    self.config.q_lora_rank, self.config.kv_lora_rank, self.config.qk_nope_head_dim,
                    self.config.qk_rope_head_dim, self.config.v_head_dim,
                    self.config.mask_size, self.config.max_cache_size, self.config.softmax_scale,
                    self.config.attention_mode.value)
            elif self.config.attention_mode == AttentionMode.PAGED_DECODE \
                or self.config.attention_mode == AttentionMode.PAGED_PREFILL:
                torch.ops.my_ops.paged_latent_attention(
                    tensors.output, tensors.Q, tensors.KV, tensors.PE,
                    tensors.WUQ, tensors.WUKV, tensors.paged_kvcache, tensors.paged_pecache,
                    tensors.cos, tensors.sin, tensors.block_tables, tensors.slots, tensors.mask, tensors.seqlen, self.config.num_heads,
                    self.config.q_lora_rank, self.config.kv_lora_rank, self.config.qk_nope_head_dim,
                    self.config.qk_rope_head_dim, self.config.v_head_dim, self.config.mask_size,
                    tensors.block_tables.shape[1], self.config.paged_block_size, self.config.softmax_scale,
                    self.config.attention_mode.value)
        return tensors.output


class MLACpu(MLA):
    def __init__(self, config: MLAConfig):
        super(MLACpu, self).__init__(config)

    def forward(self, tensors: MLATensors):
        bsz, seqlen, _ = tensors.Q.size()
        # step 1: Q upper
        if tensors.WUQ.dtype == torch.float8_e4m3fn or tensors.WUQ.dtype == torch.float8_e5m2:
            WUQ = fp8_dequant(tensors.WUQ, tensors.WUQ_scale, self.config.block_size)
            WUKV = fp8_dequant(tensors.WUKV, tensors.WUKV_scale, self.config.block_size)
        else:
            WUQ = tensors.WUQ
            WUKV = tensors.WUKV

        Q = tensors.Q.view(-1, tensors.Q.shape[-1])
        Q = torch.matmul(Q, WUQ.t())
        Q = Q.view(bsz, seqlen, self.config.num_heads, self.config.qk_nope_head_dim + self.config.qk_rope_head_dim)
        q_nope, q_pe = torch.split(Q, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1)

        cos = torch.cos(tensors.idx_theta)
        sin = torch.sin(tensors.idx_theta)
        # step 2: Q RoPE
        q_pe = q_pe.view(bsz * seqlen, self.config.num_heads, self.config.qk_rope_head_dim).contiguous()
        q_pe = apply_rope(q_pe, cos, sin)
        q_pe = q_pe.view(bsz, seqlen, self.config.num_heads, self.config.qk_rope_head_dim)

        # step 3: Q_nope * WUKV[:, :qk_nope_head_dim]
        WUKV = WUKV.view(self.config.num_heads, -1, self.config.kv_lora_rank)
        q_nope = torch.einsum("bshd, hdc->bshc", q_nope, WUKV[:, :self.config.qk_nope_head_dim])

        if self.config.attention_mode == AttentionMode.CONTINUOUS_DECODE \
            or self.config.attention_mode == AttentionMode.CONTINUOUS_PREFILL:
            kv_cache = tensors.kv_cache
            pe_cache = tensors.pe_cache
        else:
            kv_cache = torch.empty_like(tensors.kv_cache)
            pe_cache = torch.empty_like(tensors.pe_cache)
            slots = self.config.block_index_to_slots(
                tensors.block_tables,
                self.config.paged_block_size,
                self.config.context_len)
            slots = np.array(slots).reshape(self.config.batch, -1)
            # gather kv cache and pe cache
            for b in range(self.config.batch):
                kv_cache[b, :len(slots[b])] = \
                    tensors.paged_kvcache.view(-1, self.config.kv_lora_rank)[slots[b]]
                pe_cache[b, :len(slots[b])] = \
                    tensors.paged_pecache.view(-1, self.config.qk_rope_head_dim)[slots[b]]

        # step 4: QK
        max_seqlen = max(tensors.seqlen).item()
        start_pos = 0 if seqlen != 1 else max_seqlen
        end_pos = start_pos + seqlen
        kv_cache[:bsz, start_pos:end_pos, :] = tensors.KV
        PE = tensors.PE.view(bsz * seqlen, 1, self.config.qk_rope_head_dim)
        PE = apply_rope(PE, cos, sin)
        PE = PE.view(bsz, seqlen, self.config.qk_rope_head_dim)
        pe_cache[:bsz, start_pos:end_pos, :] = PE
        scores = (torch.einsum("bshc, btc->bsht", q_nope, kv_cache[:bsz, :end_pos]) +
                  torch.einsum("bshr, btr->bsht", q_pe, pe_cache[:bsz, :end_pos]))
        scores = scores * self.config.softmax_scale

        if tensors.mask is not None:
            scores += tensors.mask.unsqueeze(1)
        # step 5: softmax
        scores = scores.softmax(dim = -1, dtype = torch.float32).type_as(Q)

        # step 6: QK * KVcache
        output = torch.einsum("bsht, btc->bshc", scores, kv_cache[:bsz, :end_pos])
        # step 7: QK * WUKV[:, qk_nope_head_dim:]
        output = torch.einsum("bshc, hdc->bshd", output, WUKV[:, -self.config.v_head_dim:])
        return (output.contiguous(), kv_cache.contiguous(), pe_cache.contiguous())


def mla_decode(act_dtype: torch.dtype, weight_dtype: torch.dtype, mode: AttentionMode):
    config = MLAConfig(7, 16, 128, 64, 128, 1536, 512, 128, 192**-0.5 * 0.3,\
                       act_dtype, weight_dtype, 4096, mode)
    net_cpu = MLACpu(config)
    net_tpu = MLATpu(config)

    tensors = config.random_tensors()
    tensors_tpu = MLATensors(
        tensors.seqlen,
        tensors.WUQ.to(device),
        tensors.WUKV.to(device),
        tensors.Q.to(device),
        tensors.KV.to(device),
        tensors.PE.to(device),
        tensors.kv_cache.to(device),
        tensors.pe_cache.to(device),
        tensors.idx_theta.to(device),
        tensors.cos.to(device),
        tensors.sin.to(device),
        tensors.output.to(device),
        tensors.mask.to(device) if tensors.mask is not None else None,
        tensors.WUQ_scale.to(device) if tensors.WUQ_scale is not None else None,
        tensors.WUKV_scale.to(device) if tensors.WUKV_scale is not None else None,
        tensors.block_tables.to(device) if tensors.block_tables is not None else None,
        tensors.slots.to(device) if tensors.slots is not None else None,
        tensors.paged_kvcache.to(device) if tensors.paged_kvcache is not None else None,
        tensors.paged_pecache.to(device) if tensors.paged_pecache is not None else None
    )

    output_tpu = net_tpu(tensors_tpu)
    output_cpu, kv_cache, pe_cache = net_cpu(tensors)
    output_tpu = output_tpu.cpu()

    diff = output_tpu - output_cpu
    cosine_similarity = F.cosine_similarity(output_tpu.view(config.batch, -1),
                                            output_cpu.view(config.batch, -1), dim=-1)
    print(f"diff: {torch.max(torch.abs(diff.view(config.batch, -1)), dim=-1)}")
    print(f"cos sim: {cosine_similarity}")
    flat_cosine_similarity = F.cosine_similarity(output_tpu.flatten(), output_cpu.flatten(), dim=0)
    print(f"flat_cosine_similarity: {flat_cosine_similarity.item()}")
    if flat_cosine_similarity.item() < 0.99:
        raise RuntimeError("mla decode failed")

    # compare the kv-cache and pe-cache
    if mode == AttentionMode.CONTINUOUS_DECODE:
        kv_diff = (
            tensors_tpu.kv_cache.cpu()[:, config.context_len : config.context_len + 1]
            - kv_cache[:, config.context_len : config.context_len + 1]
        )
        pe_diff = (
            tensors_tpu.pe_cache.cpu()[:, config.context_len : config.context_len + 1]
            - pe_cache[:, config.context_len : config.context_len + 1]
        )
    else:
        kv_diff = (
            tensors_tpu.paged_kvcache.cpu().view(-1, config.kv_lora_rank)[tensors.slots]
            - kv_cache[:, config.context_len : config.context_len + 1].view(-1, config.kv_lora_rank)
        )
        pe_diff = (
            tensors_tpu.paged_pecache.cpu().view(-1, config.qk_rope_head_dim)[tensors.slots]
            - pe_cache[:, config.context_len : config.context_len + 1].view(-1, config.qk_rope_head_dim)
        )
    if torch.max(torch.abs(kv_diff)) > 1e-5:
        print(f"kv_diff: {torch.max(torch.abs(kv_diff))}")
        raise RuntimeError("mla kv-cache failed")
    if torch.max(torch.abs(pe_diff)) > 1e-5:
        print(f"pe_diff: {torch.max(torch.abs(pe_diff))}")
        raise RuntimeError("mla pe-cache failed")

def mla_prefill(act_dtype: torch.dtype, weight_dtype: torch.dtype, mode: AttentionMode):
    config = MLAConfig(16, 16, 128, 64, 128, 1536, 512, 128, 192**-0.5, act_dtype, weight_dtype, 512, mode)
    net_cpu = MLACpu(config)
    net_tpu = MLATpu(config)

    tensors = config.random_tensors()
    tensors_tpu = MLATensors(
        tensors.seqlen,
        tensors.WUQ.to(device),
        tensors.WUKV.to(device),
        tensors.Q.to(device),
        tensors.KV.to(device),
        tensors.PE.to(device),
        tensors.kv_cache.to(device),
        tensors.pe_cache.to(device),
        tensors.idx_theta.to(device),
        tensors.cos.to(device),
        tensors.sin.to(device),
        tensors.output.to(device),
        tensors.mask.to(device) if tensors.mask is not None else None,
        tensors.WUQ_scale.to(device) if tensors.WUQ_scale is not None else None,
        tensors.WUKV_scale.to(device) if tensors.WUKV_scale is not None else None,
        tensors.block_tables.to(device) if tensors.block_tables is not None else None,
        tensors.slots.to(device) if tensors.slots is not None else None,
        tensors.paged_kvcache.to(device) if tensors.paged_kvcache is not None else None,
        tensors.paged_pecache.to(device) if tensors.paged_pecache is not None else None
    )

    output_tpu = net_tpu(tensors_tpu)
    output_cpu, kv_cache, pe_cache = net_cpu(tensors)
    output_tpu = output_tpu.cpu()

    diff = output_tpu - output_cpu
    cosine_similarity = F.cosine_similarity(output_tpu.view(config.batch, -1),
                                            output_cpu.view(config.batch, -1), dim=-1)
    print(f"diff: {torch.max(torch.abs(diff.view(config.batch, -1)), dim=-1)}")
    print(f"cos sim: {cosine_similarity}")
    flat_cosine_similarity = F.cosine_similarity(output_tpu.flatten(), output_cpu.flatten(), dim=0)
    print(f"flat_cosine_similarity: {flat_cosine_similarity.item()}")
    if flat_cosine_similarity.item() < 0.99:
        raise RuntimeError("mla prefill failed")

    if mode == AttentionMode.CONTINUOUS_PREFILL:
        kv_diff = (
            tensors_tpu.kv_cache.cpu()[:, :config.context_len]
            - kv_cache[:, :config.context_len]
        )
        pe_diff = (
            tensors_tpu.pe_cache.cpu()[:, :config.context_len]
            - pe_cache[:, :config.context_len]
        )
    else:
        slots = config.block_index_to_slots(tensors.block_tables, config.paged_block_size, config.context_len)
        slots = torch.tensor(slots, dtype=torch.int64).reshape(config.batch, -1)
        kv_diff = (
            tensors_tpu.paged_kvcache.cpu().view(-1, config.kv_lora_rank)[slots]
            - kv_cache[:, :config.context_len]
        )
        pe_diff = (
            tensors_tpu.paged_pecache.cpu().view(-1, config.qk_rope_head_dim)[slots]
            - pe_cache[:, :config.context_len]
        )
    if torch.max(torch.abs(kv_diff)) > 1e-5:
        print(f"kv_diff: {torch.max(torch.abs(kv_diff))}")
        raise RuntimeError("mla kv-cache failed")
    if torch.max(torch.abs(pe_diff)) > 1e-5:
        print(f"pe_diff: {torch.max(torch.abs(pe_diff))}")
        raise RuntimeError("mla pe-cache failed")

if __name__ == "__main__":
    print(f"Test MLA Decode BF16 continuous_decode:")
    mla_decode(torch.bfloat16, torch.bfloat16, AttentionMode.CONTINUOUS_DECODE)
    print(f"----------------------------------")
    print(f"Test MLA Decode BF16 paged_decode:")
    mla_decode(torch.bfloat16, torch.bfloat16, AttentionMode.PAGED_DECODE)
    print(f"----------------------------------")
    print(f"\nTest MLA Decode FP8_E4M3 continuous_decode:")
    mla_decode(torch.bfloat16, torch.float8_e4m3fn, AttentionMode.CONTINUOUS_DECODE)
    print(f"----------------------------------")
    print(f"\nTest MLA Decode FP8_E4M3 paged_decode:")
    mla_decode(torch.bfloat16, torch.float8_e4m3fn, AttentionMode.PAGED_DECODE)
    print(f"----------------------------------")
    print(f"Test MLA Prefill BF16 continuous_prefill:")
    mla_prefill(torch.bfloat16, torch.bfloat16, AttentionMode.CONTINUOUS_PREFILL)
    print(f"----------------------------------")
    print(f"Test MLA Prefill BF16 paged_prefill:")
    mla_prefill(torch.bfloat16, torch.bfloat16, AttentionMode.PAGED_PREFILL)
    print(f"----------------------------------")
    print(f"Test MLA Prefill FP8_E4M3 continuous_prefill:")
    mla_prefill(torch.bfloat16, torch.float8_e4m3fn, AttentionMode.CONTINUOUS_PREFILL)
    print(f"----------------------------------")
    print(f"Test MLA Prefill FP8_E4M3 paged_prefill:")
    mla_prefill(torch.bfloat16, torch.float8_e4m3fn, AttentionMode.PAGED_PREFILL)
    print(f"----------------------------------")
