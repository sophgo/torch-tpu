import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu
from torch_tpu.utils.compare import cos_sim


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    使用缩放因子对量化权重进行反量化，按块处理以模拟实际的 Triton 分块计算逻辑

    Args:
        x (torch.Tensor): 量化权重张量，形状为 (M, N)
        s (torch.Tensor): 缩放因子张量，形状为 (M//block_size, N//block_size)
        block_size (int): 分块大小，默认为 128

    Returns:
        torch.Tensor: 反量化后的权重张量，与 x 形状相同
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.bfloat16)

    # 模拟 Triton 的分块计算逻辑，按块对权重进行反量化
    for pid_m in range((M + block_size - 1) // block_size):
        for pid_n in range((N + block_size - 1) // block_size):
            # 计算当前块的起始和结束索引
            offs_m_start = pid_m * block_size
            offs_m_end = min(offs_m_start + block_size, M)
            offs_n_start = pid_n * block_size
            offs_n_end = min(offs_n_start + block_size, N)

            # 提取当前块的量化权重和对应的缩放因子
            x_block = x[offs_m_start:offs_m_end, offs_n_start:offs_n_end]
            s_block = s[pid_m, pid_n]  # 每个块对应一个缩放因子

            # 反量化逻辑：将量化权重转换为float32并乘以缩放因子
            y_block = x_block.to(torch.float32) * s_block

            # 将反量化结果写回输出张量
            y[offs_m_start:offs_m_end, offs_n_start:offs_n_end] = y_block

    return y


def mm_w8a16_dq_cpu(hidden_states, weight, weight_scale, weight_block_size=128):
    """CPU 版本的矩阵乘法 W8A16 量化实现，使用精确的按块反量化逻辑"""
    # 第一步：使用按块反量化逻辑对权重进行反量化
    weight_dequantized = weight_dequant(weight, weight_scale, weight_block_size)

    # 第二步：使用反量化后的权重执行线性变换
    output = F.linear(hidden_states, weight_dequantized)

    return output.float()


def mm_w8a16_dq_tpu(
    hidden_states, weight, weight_scale, weight_block_size=128, device=None, profiler=None
):
    """使用自定义 TPU 算子的矩阵乘法 W8A16 量化实现"""
    # 第一步：将输入数据移动到指定设备
    hidden_states_tpu = hidden_states.to(device)
    weight_tpu = weight.to(device)
    weight_scale_tpu = weight_scale.to(device)
    
    # 第二步：创建输出张量
    batch_size, input_dim = hidden_states_tpu.shape
    output_dim = weight_tpu.shape[0]
    output = torch.empty(
        (batch_size, output_dim), dtype=hidden_states_tpu.dtype, device=hidden_states_tpu.device
    )

    # 第三步：调用自定义的 TPU 矩阵乘法 W8A16 量化算子
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            torch.ops.my_ops.mm_w8a16_dq_forward(
                hidden_states_tpu,
                weight_tpu,
                weight_scale_tpu,
                output,
                weight_block_size,
            )
    else:
        torch.ops.my_ops.mm_w8a16_dq_forward(
            hidden_states_tpu,
            weight_tpu,
            weight_scale_tpu,
            output,
            weight_block_size,
        )

    return output.cpu().float()


@pytest.mark.parametrize(
    "batch_size, input_dim, output_dim, weight_block_size",
    [
        (32, 7168, 1536, 128),  # 基于日志参数 q_a_proj
        (32, 7168, 576, 128),  # 基于日志参数 kv_proj
        (32, 2048, 7168, 128),  # 基于日志参数 o_proj
        (64, 4096, 2048, 128),  # 额外测试参数
    ],
)
def test_mm_w8a16_dq_forward(
    batch_size,
    input_dim,
    output_dim,
    weight_block_size,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试矩阵乘法 W8A16 量化的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        input_dim: 输入维度
        output_dim: 输出维度
        weight_block_size: 权重量化块大小
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    hidden_states = torch.randn((batch_size, input_dim), dtype=torch.bfloat16)

    # 第二步：生成权重矩阵（模拟量化后的权重）
    weight = torch.randn(output_dim, input_dim).to(torch.float8_e4m3fn)

    # 第三步：生成缩放因子
    # 根据日志中的缩放因子形状计算
    scale_height = (output_dim + weight_block_size - 1) // weight_block_size
    scale_width = (input_dim + weight_block_size - 1) // weight_block_size

    weight_scale = torch.randn((scale_height, scale_width), dtype=torch.bfloat16)
    
    # 第四步：运行 CPU 参考实现
    out_cpu = mm_w8a16_dq_cpu(
        hidden_states,
        weight,
        weight_scale,
        weight_block_size,
    )

    # 第五步：运行 TPU 实现，设备转换在mm_w8a16_dq_tpu函数内部完成
    out_tpu = mm_w8a16_dq_tpu(
        hidden_states,
        weight,
        weight_scale,
        weight_block_size,
        device=device,
        profiler=profiler,
    )

    # 第六步：比较结果
    out_tpu_cpu = out_tpu
    out_diff = out_cpu - out_tpu_cpu
    mean_diff = torch.mean(abs(out_diff)).item()

    cosm = cos_sim(out_cpu, out_tpu_cpu)
    assert cosm >= 0.99, f"Cosine similarity {cosm} is less than 0.99"
