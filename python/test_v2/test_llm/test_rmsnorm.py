import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu


def rmsnorm_cpu(x, scale=None, bias=None, axis=-1, eps=1e-8):
    """CPU 版本的 RMSNorm 函数式实现"""
    # 第一步：计算输入张量在指定轴上的平方的均值
    ms = torch.mean(torch.square(x), dim=axis, keepdim=True)
    # 第二步：计算均方根，添加epsilon防止除零
    rms = torch.sqrt(ms + eps)
    # 第三步：执行RMS归一化
    y = x / rms
    if scale is not None:
        # 第四步：应用可学习的缩放参数
        y *= scale
    if bias is not None:
        # 第五步：应用可学习的偏置参数
        y += bias
    return y


def rmsnorm_tpu(x, scale=None, bias=None, axis=-1, eps=1e-8, profiler=None):
    """使用自定义 TPU 算子的 RMSNorm 函数式实现"""
    # 第一步：创建输出张量，保持与输入相同的形状和类型
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # 第二步：调用自定义的 TPU RMSNorm 前向传播算子

    with profiler.profile(buffer_size=1024, trace_level=2):
        torch.ops.my_ops.rmsnorm_forward(x, scale, bias, output, axis, eps)
    return output


@pytest.mark.parametrize(
    "batch_size, hidden_size, axis, eps, with_scale, with_bias",
    [
        (64, 512, 3, 1e-5, True, True),
        (32, 1536, 3, 1e-5, True, True),
        (128, 1536, 3, 1e-5, False, True),
    ],
)
def test_deepseek_rmsnorm(
    batch_size,
    hidden_size,
    axis,
    eps,
    with_scale,
    with_bias,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试 RMSNorm 的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        hidden_size: 隐藏层维度
        axis: 归一化的轴
        eps: 防止除零的小常数
        with_scale: 是否使用缩放参数
        with_bias: 是否使用偏置参数
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    x = torch.randn((batch_size, 1, 1, hidden_size), requires_grad=False)
    x_tpu = x.to(device).half()

    # 第二步：准备缩放和偏置参数
    scale = None
    bias = None
    if with_scale:
        # 创建缩放参数并转换到相应设备
        scale_cpu = torch.rand(hidden_size)
        scale = (
            scale_cpu.clone()
            .detach()
            .contiguous()
            .requires_grad_(False)
            .to(device)
            .half()
        )
    if with_bias:
        # 创建偏置参数并转换到相应设备
        bias_cpu = torch.rand(hidden_size)
        bias = (
            bias_cpu.clone()
            .detach()
            .contiguous()
            .requires_grad_(False)
            .to(device)
            .half()
        )

    # 第三步：运行 CPU 参考实现
    scale_cpu_for_calc = scale_cpu if with_scale else None
    bias_cpu_for_calc = bias_cpu if with_bias else None
    out_cpu = rmsnorm_cpu(
        x, scale=scale_cpu_for_calc, bias=bias_cpu_for_calc, axis=axis, eps=eps
    )

    # 第四步：运行 TPU 实现，在这里进行性能分析
    # 在 profiler 上下文中执行 TPU RMSNorm 计算
    out_tpu = rmsnorm_tpu(
        x_tpu, scale=scale, bias=bias, axis=axis, eps=eps, profiler=profiler
    )

    out_tpu = out_tpu.float().cpu().detach()

    # 第五步：计算差异并验证结果
    out_diff = out_cpu - out_tpu
    mean_diff = torch.mean(abs(out_diff)).item()

    # 由于使用了半精度计算，允许较大的误差
    tolerance = 0.1

    # 断言平均差异在容忍范围内
    assert (
        mean_diff < tolerance
    ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"
