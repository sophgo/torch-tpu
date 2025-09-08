import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu


def cos_sim(vector_a, vector_b):
    """计算两个向量的余弦相似度"""
    vector_a = vector_a.reshape(-1)
    vector_b = vector_b.reshape(-1)
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    with np.errstate(invalid="ignore"):
        cos = np.nan_to_num(num / denom)
    sim = 0.5 + 0.5 * cos
    return sim


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    使用提供的缩放张量对给定的权重张量进行反量化

    Args:
        x (torch.Tensor): 形状为 (M, N) 的量化权重张量
        s (torch.Tensor): 形状为 (M, N) 的缩放张量
        block_size (int, optional): 分块大小。默认为 128

    Returns:
        torch.Tensor: 与 `x` 形状相同的反量化权重张量
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"

    M, N = x.size()
    # 第一步：创建输出张量，使用默认数据类型
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    # 第二步：模拟 Triton 的分块计算逻辑，遍历每个块
    for pid_m in range((M + block_size - 1) // block_size):
        for pid_n in range((N + block_size - 1) // block_size):
            # 第三步：计算当前块的起始和结束索引
            offs_m_start = pid_m * block_size
            offs_m_end = min(offs_m_start + block_size, M)
            offs_n_start = pid_n * block_size
            offs_n_end = min(offs_n_start + block_size, N)

            # 第四步：提取当前块的量化权重和缩放因子
            x_block = x[offs_m_start:offs_m_end, offs_n_start:offs_n_end]
            s_block = s[pid_m, pid_n]  # 假设 s 的每个块对应一个缩放因子

            # 第五步：反量化逻辑，将量化权重乘以缩放因子
            y_block = x_block.to(torch.float32) * s_block

            # 第六步：将反量化结果写回输出张量
            y[offs_m_start:offs_m_end, offs_n_start:offs_n_end] = y_block

    return y


def deepseek_mm_cpu(x, w0, s0, block_size):
    """CPU 版本的 Deepseek MM 函数式实现"""
    # 第一步：对权重进行反量化
    dequantized_weight0 = weight_dequant(w0, s0, block_size)
    # 第二步：执行线性变换
    out = F.linear(x, dequantized_weight0)
    return out


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
    out_cpu = deepseek_mm_cpu(x, w0, s0, block_size)

    # 第七步：运行 TPU 实现，在 profiler 上下文中执行
    out_tpu = deepseek_mm_tpu(x_tpu, w0_tpu, s0_tpu, block_size, profiler)

    # 第八步：使用比较函数进行结果验证
    metrics = cmp_deepseek_mm(out_cpu, out_tpu, tolerance=0.1)

    # 第九步：断言验证结果在容差范围内
    assert (
        metrics["mean_diff"] < 0.1
    ), f"Mean difference {metrics['mean_diff']} exceeds tolerance 0.1"
    assert (
        metrics["cosine_sim"] > 0.9
    ), f"Cosine similarity {metrics['cosine_sim']} is too low"


@pytest.mark.parametrize(
    "batch_size, input_w, middle_w",
    [
        (128, 7168, 1536),
        (256, 4096, 1024),
    ],
)
@pytest.mark.priority_medium
def test_deepseek_mm_large_batch(
    batch_size,
    input_w,
    middle_w,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试大批次大小的 Deepseek MM 实现
    """
    block_size = 128
    test_deepseek_mm(
        batch_size,
        input_w,
        middle_w,
        block_size,
        device,
        setup_random_seed,
        profiler,
    )


@pytest.mark.parametrize("block_size", [64, 128, 256])
@pytest.mark.priority_low
def test_deepseek_mm_block_sizes(
    block_size,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试不同分块大小的 Deepseek MM 实现
    """
    batch_size = 32
    input_w = 2048
    middle_w = 512

    test_deepseek_mm(
        batch_size,
        input_w,
        middle_w,
        block_size,
        device,
        setup_random_seed,
        profiler,
    )
