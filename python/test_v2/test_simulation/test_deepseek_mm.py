import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu


def deepseek_mm_tpu(x, w0, s0, block_size, profiler=None):
    """使用自定义 TPU 算子的 Deepseek MM 函数式实现"""
    # 第一步：创建输出张量，形状为 (batch_size, 输出特征数)
    output = torch.empty((x.shape[0], w0.shape[0]), dtype=x.dtype, device=x.device)

    # 第二步：在性能分析器上下文中调用自定义的 TPU MM 前向传播算子
    with profiler.profile(buffer_size=1024, trace_level=2):
        torch.ops.my_ops.mm_w8a16_dq_forward(x, w0, s0, output, block_size)

    return output


def cmp_deepseek_mm(out_cpu, out_tpu, tolerance=0.1):
    """
    比较 CPU 和 TPU 输出结果的函数

    Args:
        out_cpu: CPU 输出结果
        out_tpu: TPU 输出结果
        tolerance: 容差值

    Returns:
        dict: 包含各种比较指标的字典
    """
    # 第一步：将输出转换为相同格式进行比较
    out_cpu_flat = out_cpu.float().flatten()
    out_tpu_flat = out_tpu.float().cpu().flatten()

    # 第二步：计算差异
    out_diff = out_cpu_flat - out_tpu_flat
    ratio = abs(out_diff / torch.max(abs(out_cpu_flat), abs(out_tpu_flat)))

    # 第三步：找到最大差异的位置
    max_ratio_idx = torch.argmax(ratio)

    # 第四步：计算余弦相似度
    cosine_sim = cos_sim(out_cpu_flat.numpy(), out_tpu_flat.numpy())

    # 第五步：计算平均差异
    mean_diff = torch.mean(abs(out_diff)).item()

    return {
        "mean_diff": mean_diff,
        "max_ratio": ratio[max_ratio_idx].item(),
        "cosine_sim": cosine_sim,
        "max_ratio_cpu_val": out_cpu_flat[max_ratio_idx].item(),
        "max_ratio_tpu_val": out_tpu_flat[max_ratio_idx].item(),
    }


@pytest.mark.parametrize(
    "batch_size, input_w, middle_w, block_size",
    [
        (32, 7168, 1536, 128),
        (64, 4096, 1024, 128),
        (16, 2048, 512, 64),
    ],
)
@pytest.mark.priority_high
def test_deepseek_mm(
    batch_size,
    input_w,
    middle_w,
    block_size,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试 Deepseek MM 的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        input_w: 输入特征维度
        middle_w: 输出特征维度
        block_size: 分块大小
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：计算缩放因子的维度
    scale_m = (middle_w + block_size - 1) // block_size
    scale_n = (input_w + block_size - 1) // block_size

    # 第二步：设置默认数据类型为 bfloat16
    torch.set_default_dtype(torch.bfloat16)

    # 第三步：生成测试输入数据
    x = torch.randn(batch_size, input_w)  # 输入数据
    w0 = torch.randn(middle_w, input_w)  # 量化权重
    s0 = torch.rand(scale_m, scale_n)  # 缩放因子矩阵

    # 第四步：将权重转换为 float8_e4m3fn 格式
    w0 = w0.to(torch.float8_e4m3fn)

    # 第五步：将数据转移到 TPU 设备
    x_tpu = x.to(device)
    w0_tpu = w0.to(device)
    s0_tpu = s0.to(device)

    # 第六步：运行 CPU 参考实现

    # 第七步：运行 TPU 实现，在 profiler 上下文中执行
    out_tpu = deepseek_mm_tpu(x_tpu, w0_tpu, s0_tpu, block_size, profiler)
    from torch_tpu.utils.tensor_like import graph
    breakpoint() 