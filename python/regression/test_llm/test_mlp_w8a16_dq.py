import pdb
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
    权重反量化函数，参考 deepseek_mlp.py 的实现

    Args:
        x (torch.Tensor): 量化权重张量，形状为 (M, N)
        s (torch.Tensor): 缩放因子张量，形状为 (M/block_size, N/block_size)
        block_size (int): 块大小，默认为 128

    Returns:
        torch.Tensor: 反量化后的权重张量，形状与 x 相同
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.bfloat16)  # 输出张量

    # 第一步：按块进行反量化计算，模拟 Triton 的分块计算逻辑
    for pid_m in range((M + block_size - 1) // block_size):
        for pid_n in range((N + block_size - 1) // block_size):
            # 第二步：计算当前块的起始和结束索引
            offs_m_start = pid_m * block_size
            offs_m_end = min(offs_m_start + block_size, M)
            offs_n_start = pid_n * block_size
            offs_n_end = min(offs_n_start + block_size, N)

            # 第三步：提取当前块的量化权重和对应的缩放因子
            x_block = x[offs_m_start:offs_m_end, offs_n_start:offs_n_end]
            s_block = s[pid_m, pid_n]  # 每个块对应一个缩放因子

            # 第四步：执行反量化，将量化权重乘以缩放因子
            y_block = x_block.to(torch.float32) * s_block

            # 第五步：将反量化结果写回输出张量
            y[offs_m_start:offs_m_end, offs_n_start:offs_n_end] = y_block

    return y


def mlp_w8a16_dq_cpu(
    x,
    gate_weight,
    up_weight,
    down_weight,
    gate_scale,
    up_scale,
    down_scale,
    blocksize=128,
):
    """CPU 版本的 MLP W8A16 量化前向传播实现，参考 deepseek_mlp.py"""

    # 第一步：对 gate_weight 进行反量化，然后做线性变换并应用 SiLU 激活
    dequantized_gate_weight = weight_dequant(gate_weight, gate_scale, blocksize)
    gate_out = F.linear(x, dequantized_gate_weight)
    gate_activated = F.silu(gate_out)  # SiLU 激活函数

    # 第二步：对 up_weight 进行反量化，然后做线性变换
    dequantized_up_weight = weight_dequant(up_weight, up_scale, blocksize)
    up_out = F.linear(x, dequantized_up_weight)

    # 第三步：执行门控机制，将 gate 分支和 up 分支的结果相乘
    gated_out = gate_activated * up_out

    # 第四步：对 down_weight 进行反量化，然后做最终的线性变换
    dequantized_down_weight = weight_dequant(down_weight, down_scale, blocksize)
    output = F.linear(gated_out, dequantized_down_weight)

    return output.float().flatten()


@torch.no_grad()
def mlp_w8a16_dq_tpu(
    x,
    gate_weight,
    up_weight,
    down_weight,
    gate_scale,
    up_scale,
    down_scale,
    blocksize=128,
    device=None,
    profiler=None,
):
    """使用自定义 TPU 算子的 MLP W8A16 量化前向传播实现"""
    # 第一步：将所有输入数据移动到指定设备
    x_tpu = x.to(device)
    gate_weight_tpu = gate_weight.to(device)
    up_weight_tpu = up_weight.to(device)
    down_weight_tpu = down_weight.T.to(device)  # 转置用于 TPU
    gate_scale_tpu = gate_scale.to(device)
    up_scale_tpu = up_scale.to(device)
    down_scale_tpu = down_scale.T.to(device)  # 转置用于 TPU
    
    # 第二步：创建输出张量，MLP 输出维度与输入的最后一维相同
    batch_size = x_tpu.shape[0]
    hidden_size = x_tpu.shape[1]
    output = torch.empty((batch_size, hidden_size), dtype=x_tpu.dtype, device=x_tpu.device)

    # 第三步：调用自定义的 TPU MLP W8A16 量化前向传播算子
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            torch.ops.my_ops.mlp_w8a16_dq_forward(
                x_tpu,
                gate_weight_tpu,
                up_weight_tpu,
                down_weight_tpu,
                gate_scale_tpu,
                up_scale_tpu,
                down_scale_tpu,
                output,
                blocksize,
            )
    else:
        torch.ops.my_ops.mlp_w8a16_dq_forward(
            x_tpu,
            gate_weight_tpu,
            up_weight_tpu,
            down_weight_tpu,
            gate_scale_tpu,
            up_scale_tpu,
            down_scale_tpu,
            output,
            blocksize,
        )

    return output.cpu().float().flatten()


@pytest.mark.parametrize(
    "batch_size, hidden_size, intermediate_size, blocksize",
    [
        (32, 7168, 2304, 128),  # 基于日志参数
        # (32, 7168, 256, 128),  # 基于日志参数
        # (64, 4096, 2048, 128),  # 额外测试参数
    ],
)
def test_mlp_w8a16_dq_forward(
    batch_size,
    hidden_size,
    intermediate_size,
    blocksize,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试 MLP W8A16 量化前向传播的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        hidden_size: 隐藏层维度
        intermediate_size: 中间层维度
        blocksize: 量化块大小
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    x = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16)

    # 第二步：生成权重矩阵（模拟量化后的权重），确保 contiguous
    gate_weight = torch.randn(intermediate_size, hidden_size).to(torch.float8_e4m3fn)
    up_weight = torch.randn(intermediate_size, hidden_size).to(torch.float8_e4m3fn)
    down_weight = torch.randn(hidden_size, intermediate_size).to(torch.float8_e4m3fn)

    # 第三步：生成缩放因子，确保 contiguous
    # 根据日志中的缩放因子形状计算
    scale_height = (intermediate_size + blocksize - 1) // blocksize
    scale_width = (hidden_size + blocksize - 1) // blocksize

    gate_scale = torch.randn((scale_height, scale_width), dtype=torch.bfloat16)
    up_scale = torch.randn((scale_height, scale_width), dtype=torch.bfloat16)
    down_scale = torch.randn((scale_width, scale_height), dtype=torch.bfloat16)
    
    # 第四步：运行 CPU 参考实现
    # 注意：CPU 使用原始形状的权重，不需要转置
    out_cpu = mlp_w8a16_dq_cpu(
        x,
        gate_weight,
        up_weight,
        down_weight,  # CPU 使用原始形状的 down_weight
        gate_scale,
        up_scale,
        down_scale,  # CPU 使用原始形状的 down_scale
        blocksize,
    )

    # 第五步：运行 TPU 实现，设备转换在mlp_w8a16_dq_tpu函数内部完成
    out_tpu = mlp_w8a16_dq_tpu(
        x,
        gate_weight,
        up_weight,
        down_weight,
        gate_scale,
        up_scale,
        down_scale,
        blocksize,
        device=device,
        profiler=profiler,
    )

    # 第六步：比较结果
    out_tpu_cpu = out_tpu
    out_diff = out_cpu - out_tpu_cpu

    csim = cos_sim(out_cpu, out_tpu_cpu)
    assert csim >= 0.99, f"Cosine similarity {csim} is less than 0.99"
