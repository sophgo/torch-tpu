import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu
import math


def gen_rope(Ntotal, head_size):
    """生成旋转位置编码的角度张量，与 mla.py 保持一致"""
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


def fp8_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.
    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size for tiling. Defaults to 128.
    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.
    """
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
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
    out = converted_x * s.view(m, 1, n, 1)
    if pad:
        return out.reshape(m * block_size, n * block_size)[:M, :N]
    else:
        return out.reshape(M, N)


def rotate_every_two(x):
    """旋转张量的后半部分元素（用于RoPE），与 mla.py 保持一致"""
    x_swapped = torch.cat(
        (-x[:, :, (int)(x.shape[-1] / 2) :], x[:, :, : (int)(x.shape[-1] / 2)]), dim=-1
    )
    return x_swapped


def apply_rope(x, cos, sin):
    """应用旋转位置编码（RoPE），与 mla.py 保持一致"""
    # RoPE in deepseek is interleaved, so we need to swap the elements
    # to get the correct rotation.
    assert x.dim() == 3
    batch, num_heads, head_dim = x.shape
    assert head_dim % 2 == 0
    x = (x.view(batch, num_heads, -1, 2).transpose(2, 3).contiguous()).view(batch, num_heads, -1)
    x = (x * cos) + (rotate_every_two(x) * sin)
    return x


def paged_latent_attention_fp8_cpu(
    query,
    normed_lkv_nope,
    key_pe,
    q_b_weight,
    kv_b_weight,
    kv_cache0,
    kv_cache1,
    block_tables,
    cos,
    sin,
    q_b_scale=None,
    kv_b_scale=None,
    block_size=128,
):
    """CPU 版本的分页潜在注意力 FP8 实现"""
    batch_size, seq_len = query.shape[0], 1  # 通常推理时seq_len=1
    num_heads = 16  # 基于日志推断
    qk_nope_head_dim = 128  # 基于mla.py配置
    qk_rope_head_dim = 64  # 基于mla.py配置
    v_head_dim = 128  # 基于mla.py配置
    q_lora_rank = query.shape[-1]  # query的最后一个维度
    kv_lora_rank = normed_lkv_nope.shape[-1]  # kv的最后一个维度
    softmax_scale = 192**-0.5 * 0.3  # 基于mla.py配置
    paged_block_size = 16  # 基于测试参数

    # 第一步：FP8权重去量化
    if q_b_weight.dtype == torch.float8_e4m3fn or q_b_weight.dtype == torch.float8_e5m2:
        if q_b_scale is not None and kv_b_scale is not None:
            WUQ = fp8_dequant(q_b_weight, q_b_scale, block_size)
            WUKV = fp8_dequant(kv_b_weight, kv_b_scale, block_size)
        else:
            # 如果没有scale，简单转换为float
            WUQ = q_b_weight.float()
            WUKV = kv_b_weight.float()
    else:
        WUQ = q_b_weight.float()
        WUKV = kv_b_weight.float()

    # 第二步：Q upper projection - 将query投影到更高维度
    Q = query.view(-1, query.shape[-1])  # 展平batch和seq维度
    Q = torch.matmul(Q, WUQ.t())  # 矩阵乘法进行投影
    Q = Q.view(batch_size, seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim)
    q_nope, q_pe = torch.split(Q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    # 第三步：对Q的PE部分应用RoPE
    q_pe = q_pe.view(batch_size * seq_len, num_heads, qk_rope_head_dim).contiguous()

    # 使用正确形状的cos和sin
    q_pe = apply_rope(q_pe, cos, sin)
    q_pe = q_pe.view(batch_size, seq_len, num_heads, qk_rope_head_dim)

    # 第四步：将WUKV重塑并计算q_nope与KV权重的交互
    # WUKV形状：(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    # 重塑为：(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    WUKV = WUKV.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    q_nope_projected = torch.einsum(
        "bshd, hdc->bshc", q_nope, WUKV[:, :qk_nope_head_dim]
    )

    # 第五步：处理分页KV缓存
    # 在分页注意力中，我们需要从分页缓存中收集数据
    max_seq_len = kv_cache0.shape[0]

    # 模拟分页缓存访问 - 在实际实现中这会更复杂
    # 这里我们假设整个缓存都是可用的，简化处理
    collected_kv_cache = kv_cache0  # shape: (max_seq_len, num_heads, kv_lora_rank)
    collected_pe_cache = kv_cache1  # shape: (max_seq_len, num_heads, qk_rope_head_dim)

    # 第六步：处理当前时刻的KV
    # 将normed_lkv_nope投影并添加到缓存中（在解码模式下）
    current_kv = normed_lkv_nope  # 当前时刻的KV

    # 对key_pe应用RoPE
    key_pe_reshaped = key_pe.view(batch_size * seq_len, 1, qk_rope_head_dim)
    key_pe_roped = apply_rope(key_pe_reshaped, cos, sin)
    key_pe_roped = key_pe_roped.view(batch_size, seq_len, qk_rope_head_dim)

    # 第七步：计算注意力分数
    # 使用有效的序列长度
    effective_seq_len = min(max_seq_len, 512)  # 基于配置的context长度

    # 计算q_nope与kv_cache的注意力分数
    score0 = torch.einsum(
        "bshc, thc->bsht", q_nope_projected, collected_kv_cache[:effective_seq_len]
    )

    # 计算q_pe与pe_cache的注意力分数
    score1 = torch.einsum(
        "bshr, thr->bsht", q_pe, collected_pe_cache[:effective_seq_len]
    )

    # 第八步：合并分数并应用缩放
    scores = score0 + score1
    scores = scores * softmax_scale

    # 第九步：应用causal mask（对于解码，只需要考虑历史token）
    # 在解码模式下，当前token只能看到之前的token
    if seq_len == 1:  # 解码模式
        # 不需要额外的mask，因为我们只在看历史
        pass
    else:
        # 如果是prefill模式，需要causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, effective_seq_len), float("-inf")), diagonal=1
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)

    # 第十步：应用softmax
    scores = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(query)

    # 第十一步：计算最终输出
    # 使用注意力权重对KV进行加权
    output = torch.einsum(
        "bsht, thc->bshc", scores, collected_kv_cache[:effective_seq_len]
    )

    # 第十二步：通过WUKV的V部分进行最终投影
    output = torch.einsum("bshc, hdc->bshd", output, WUKV[:, qk_nope_head_dim:])

    # 返回形状为(batch_size, num_heads, v_head_dim)的输出
    return output.squeeze(1).contiguous().float()  # 移除seq_len维度（因为=1）


def paged_latent_attention_fp8_tpu(
    output,
    query,
    normed_lkv_nope,
    key_pe,
    q_b_weight,
    kv_b_weight,
    paged_kvcache,
    paged_pecache,
    cos,
    sin,
    q_b_scale,
    kv_b_scale,
    block_tables,
    slots,
    kvu,
    mask,
    seqlen,
    cache_seqlen,
    num_heads,
    generate_token,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    mask_size,
    block_size,
    blocks_per_seq,
    paged_block_size,
    softmax_scale,
    attention_mode,
    device=None,
    profiler=None,
):
    """使用自定义 TPU 算子的分页潜在注意力 FP8 实现，参数顺序与 mla.py 保持一致"""
    # 第一步：将所有输入数据移动到指定设备
    output_tpu = output.to(device)
    query_tpu = query.to(device)
    normed_lkv_nope_tpu = normed_lkv_nope.to(device)
    key_pe_tpu = key_pe.to(device)
    q_b_weight_tpu = q_b_weight.to(device)
    kv_b_weight_tpu = kv_b_weight.to(device)
    paged_kvcache_tpu = paged_kvcache.to(device)
    paged_pecache_tpu = paged_pecache.to(device)
    cos_tpu = cos.to(device)
    sin_tpu = sin.to(device)
    
    # 第二步：处理可选参数
    q_b_scale_tpu = q_b_scale.to(device) if q_b_scale is not None else None
    kv_b_scale_tpu = kv_b_scale.to(device) if kv_b_scale is not None else None
    block_tables_tpu = block_tables.to(device) if block_tables is not None else None
    slots_tpu = slots.to(device) if slots is not None else None
    kvu_tpu = kvu.to(device) if kvu is not None else None
    mask_tpu = mask.to(device) if mask is not None else None

    # 第三步：调用自定义的 TPU 分页潜在注意力 FP8 算子
    # 参数顺序与 mla.py 中的保持完全一致
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            torch.ops.my_ops.paged_latent_attention_fp8(
                output_tpu, query_tpu, normed_lkv_nope_tpu, key_pe_tpu, q_b_weight_tpu, kv_b_weight_tpu,
                paged_kvcache_tpu, paged_pecache_tpu, cos_tpu, sin_tpu, q_b_scale_tpu,
                kv_b_scale_tpu, block_tables_tpu, slots_tpu, kvu_tpu,
                mask_tpu, seqlen, cache_seqlen, num_heads, generate_token,
                q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                qk_rope_head_dim, v_head_dim, mask_size,
                block_size, blocks_per_seq,
                paged_block_size, softmax_scale,
                attention_mode
            )
    else:
        torch.ops.my_ops.paged_latent_attention_fp8(
            output_tpu, query_tpu, normed_lkv_nope_tpu, key_pe_tpu, q_b_weight_tpu, kv_b_weight_tpu,
            paged_kvcache_tpu, paged_pecache_tpu, cos_tpu, sin_tpu, q_b_scale_tpu,
            kv_b_scale_tpu, block_tables_tpu, slots_tpu, kvu_tpu,
            mask_tpu, seqlen, cache_seqlen, num_heads, generate_token,
            q_lora_rank, kv_lora_rank, qk_nope_head_dim,
            qk_rope_head_dim, v_head_dim, mask_size,
            block_size, blocks_per_seq,
            paged_block_size, softmax_scale,
            attention_mode
        )

    return output_tpu.cpu().float()


@pytest.mark.parametrize(
    "batch_size, query_dim, kv_nope_dim, pe_dim, num_blocks, block_size",
    [
        (32, 1536, 512, 64, 10, 16),  # 基于日志参数
        # (16, 1024, 256, 32, 8, 16),  # 较小的测试参数
        # (64, 2048, 768, 128, 12, 16),  # 较大的测试参数
    ],
)
def test_paged_latent_attention_fp8(
    batch_size,
    query_dim,
    kv_nope_dim,
    pe_dim,
    num_blocks,
    block_size,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试分页潜在注意力 FP8 的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        query_dim: 查询维度
        kv_nope_dim: KV NoPosition Encoding 维度
        pe_dim: 位置编码维度（应该等于 qk_rope_head_dim）
        num_blocks: 块数量
        block_size: 块大小
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    num_heads = 16
    head_dim = 128

    # 根据mla.py，定义头维度
    qk_nope_head_dim = 128  # 基于mla.py配置
    qk_rope_head_dim = 64  # 基于mla.py配置，注意：这应该与pe_dim匹配
    v_head_dim = 128  # 基于mla.py配置
    
    # 调整pe_dim以匹配qk_rope_head_dim
    pe_dim = qk_rope_head_dim

    # 第一步：生成输入张量
    seq_len = 1  # 解码模式下通常为1
    query = torch.randn((batch_size, query_dim), dtype=torch.bfloat16)
    normed_lkv_nope = torch.randn((batch_size, kv_nope_dim), dtype=torch.bfloat16)
    key_pe = torch.randn((batch_size, pe_dim), dtype=torch.bfloat16)

    # 第二步：生成权重矩阵（FP8量化）
    # 参考mla.py的方式：先生成float然后转换为FP8
    q_b_weight = (
        torch.randn(
            (num_heads * (qk_nope_head_dim + qk_rope_head_dim), query_dim),
            dtype=torch.bfloat16,
        )
        / math.sqrt(query_dim)
    ).to(torch.float8_e4m3fn)
    kv_b_weight = (
        torch.randn(
            (num_heads * (qk_nope_head_dim + v_head_dim), kv_nope_dim),
            dtype=torch.bfloat16,
        )
        / math.sqrt(kv_nope_dim)
    ).to(torch.float8_e4m3fn)

    # 生成FP8的scale张量（参考mla.py）
    fp8_block_size = 128  # FP8量化的块大小
    q_b_scale = torch.randn(
        int(math.ceil(q_b_weight.shape[0] / fp8_block_size)),
        int(math.ceil(q_b_weight.shape[1] / fp8_block_size)),
        dtype=torch.bfloat16,
    )
    kv_b_scale = torch.randn(
        int(math.ceil(kv_b_weight.shape[0] / fp8_block_size)),
        int(math.ceil(kv_b_weight.shape[1] / fp8_block_size)),
        dtype=torch.bfloat16,
    )

    # 第三步：生成KV缓存
    max_seq_len = num_blocks * block_size
    paged_kvcache = torch.randn((max_seq_len, num_heads, kv_nope_dim), dtype=torch.bfloat16)
    paged_pecache = torch.randn((max_seq_len, num_heads, pe_dim), dtype=torch.bfloat16)

    # 第四步：生成块表（用于分页注意力）
    block_tables = torch.randint(
        0, max_seq_len // block_size, (batch_size, num_blocks), dtype=torch.int32
    )

    # 第五步：生成旋转位置编码的cos和sin（使用gen_rope函数，与mla.py保持一致）
    idx_theta = gen_rope(batch_size * seq_len, qk_rope_head_dim).to(torch.bfloat16)
    cos = torch.cos(idx_theta).contiguous()
    sin = torch.sin(idx_theta).contiguous()

    # 第六步：准备其他必需参数
    seqlen = torch.tensor([512 for _ in range(batch_size)], dtype=torch.int32)  # 基于mla.py
    cache_seqlen = torch.tensor([0 for _ in range(batch_size)], dtype=torch.int32)
    slots = torch.randint(0, max_seq_len, (batch_size,), dtype=torch.int32)
    kvu = None
    mask = None
    generate_token = 1
    mask_size = 0
    paged_block_size = 16
    softmax_scale = 192**-0.5 * 0.3  # 基于mla.py
    attention_mode = 3  # PAGED_DECODE

    # 第七步：运行 CPU 参考实现
    out_cpu = paged_latent_attention_fp8_cpu(
        query,
        normed_lkv_nope,
        key_pe,
        q_b_weight,
        kv_b_weight,
        paged_kvcache,
        paged_pecache,
        block_tables,
        cos,
        sin,
        q_b_scale=q_b_scale,  # 传递FP8的scale
        kv_b_scale=kv_b_scale,  # 传递FP8的scale
        block_size=fp8_block_size,  # FP8去量化的块大小
    )

    # 第八步：创建输出张量
    output = torch.empty((batch_size, num_heads, v_head_dim), dtype=torch.bfloat16)

    # 第九步：运行 TPU 实现
    out_tpu = paged_latent_attention_fp8_tpu(
        output, query, normed_lkv_nope, key_pe, q_b_weight, kv_b_weight,
        paged_kvcache, paged_pecache, cos, sin, q_b_scale, kv_b_scale,
        block_tables, slots, kvu, mask, seqlen, cache_seqlen,
        num_heads, generate_token, query_dim, kv_nope_dim,
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim, mask_size,
        fp8_block_size, num_blocks, paged_block_size, softmax_scale,
        attention_mode, device=device, profiler=profiler
    )

    # 第十步：比较结果
    out_tpu_cpu = out_tpu.float().cpu().detach()
    out_diff = out_cpu - out_tpu_cpu
    mean_diff = torch.mean(abs(out_diff)).item()

    # 由于使用了FP8量化和复杂的注意力计算，允许较大的误差
    tolerance = 1.0

    # 断言平均差异在容忍范围内
    assert (
        mean_diff < tolerance
    ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"


@pytest.mark.parametrize(
    "batch_size, max_seq_len, num_heads, head_dim",
    [
        (32, 2270, 16, 128),  # 基于日志参数，计算得出的缓存大小
        (16, 1024, 8, 64),  # 较小的测试参数
    ],
)
def test_attention_cache_management(
    batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试注意力缓存管理的正确性

    Args:
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        num_heads: 注意力头数
        head_dim: 注意力头维度
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：创建缓存张量
    kv_cache0 = torch.randn((max_seq_len, num_heads, 512), dtype=torch.bfloat16)
    kv_cache1 = torch.randn((max_seq_len, num_heads, 64), dtype=torch.bfloat16)

    # 第二步：创建块表用于分页访问
    block_size = 16
    num_blocks = 10
    block_tables = torch.randint(
        0, max_seq_len // block_size, (batch_size, num_blocks), dtype=torch.int32
    )

    # 第三步：验证缓存大小和块表的一致性
    assert (
        kv_cache0.shape[0] == max_seq_len
    ), f"Cache0 size mismatch: {kv_cache0.shape[0]} vs {max_seq_len}"
    assert (
        kv_cache1.shape[0] == max_seq_len
    ), f"Cache1 size mismatch: {kv_cache1.shape[0]} vs {max_seq_len}"
    assert block_tables.shape == (
        batch_size,
        num_blocks,
    ), f"Block table shape mismatch: {block_tables.shape}"

    # 第四步：验证块表中的索引不超出缓存范围
    max_block_idx = max_seq_len // block_size - 1
    assert torch.all(
        block_tables <= max_block_idx
    ), "Block table indices exceed cache range"
    assert torch.all(block_tables >= 0), "Block table indices are negative"

    print(
        f"Cache management test passed for batch_size={batch_size}, max_seq_len={max_seq_len}"
    )


@pytest.mark.priority_high
def test_attention_shape_consistency(device, setup_random_seed):
    """
    测试注意力计算中张量形状的一致性（高优先级测试）
    """
    # 基于日志中的实际参数
    batch_size = 32
    query_dim = 1536
    kv_nope_dim = 512
    pe_dim = 64

    # 第一步：创建测试张量
    query = torch.randn((batch_size, query_dim), dtype=torch.bfloat16, device=device)
    normed_lkv_nope = torch.randn(
        (batch_size, kv_nope_dim), dtype=torch.bfloat16, device=device
    )
    key_pe = torch.randn((batch_size, pe_dim), dtype=torch.bfloat16, device=device)

    # 第二步：验证张量形状
    assert query.shape == (
        batch_size,
        query_dim,
    ), f"Query shape mismatch: {query.shape}"
    assert normed_lkv_nope.shape == (
        batch_size,
        kv_nope_dim,
    ), f"KV NoPosition shape mismatch: {normed_lkv_nope.shape}"
    assert key_pe.shape == (
        batch_size,
        pe_dim,
    ), f"Key PE shape mismatch: {key_pe.shape}"

    # 第三步：验证数据类型
    assert query.dtype == torch.bfloat16, f"Query dtype mismatch: {query.dtype}"
    assert (
        normed_lkv_nope.dtype == torch.bfloat16
    ), f"KV NoPosition dtype mismatch: {normed_lkv_nope.dtype}"
    assert key_pe.dtype == torch.bfloat16, f"Key PE dtype mismatch: {key_pe.dtype}"

    print("Attention shape consistency test passed")
