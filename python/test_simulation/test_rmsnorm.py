import torch
from torch_tpu.utils.reflection.graph import print_graph_summary

import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu
import ctypes as ct


def rmsnorm_tpu(x, scale=None, bias=None, axis=-1, eps=1e-8):
    """使用自定义 TPU 算子的 RMSNorm 函数式实现"""
    # 第一步：创建输出张量，保持与输入相同的形状和类型
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # 第二步：调用自定义的 TPU RMSNorm 前向传播算子
    # os.environ['FORBID_CMD_EXECUTE'] = "1"
    torch.ops.my_ops.set_env("FILE_DUMP_CMD", "rmsnorm/ins")
    os.makedirs("rmsnorm", exist_ok=True)

    torch.ops.my_ops.rmsnorm_forward(x, scale, bias, output, axis, eps)
    return output


def rmsnorm_tpu2(x, scale=None, bias=None, axis=-1, eps=1e-8):
    """使用自定义 TPU 算子的 RMSNorm 函数式实现"""
    # 第一步：创建输出张量，保持与输入相同的形状和类型
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # 第二步：调用自定义的 TPU RMSNorm 前向传播算子
    # os.environ['FORBID_CMD_EXECUTE'] = "1"
    torch.ops.my_ops.set_env("FILE_DUMP_CMD", "rmsnorm2/ins")
    os.makedirs("rmsnorm2", exist_ok=True)
    torch.ops.my_ops.rmsnorm_forward(x, scale, bias, output, axis, eps)
    return output


@pytest.mark.parametrize(
    "batch_size, hidden_size, axis, eps, with_scale, with_bias",
    [
        (256, 7168, 3, 1e-5, True, True),
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

    # 第四步：运行 TPU 实现，在这里进行性能分析
    # 在 profiler 上下文中执行 TPU RMSNorm 计算
    out_tpu = rmsnorm_tpu(x_tpu, scale=scale, bias=bias, axis=axis, eps=eps)
    out_tpu2 = rmsnorm_tpu2(x_tpu, scale=scale, bias=bias, axis=axis, eps=eps)
    x_tpu[0] = 1

    out_tpu = out_tpu.float().cpu().detach()
    # out_tpu2 = out_tpu2.float().cpu().detach()

    print_graph_summary()

    # 第五步：计算差异并验证结果
