import torch
from torch_tpu.utils.reflection.graph import print_graph_summary

import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu


def matmal_w4a16_tpu(x, q_weight, q_scale_zp, bias=None, group_size=128, weight_bits=4):
    """使用自定义 TPU 算子的 w4a16 matmul 函数式实现"""
    # 第一步：创建输出张量
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    torch.ops.my_ops.set_env("FILE_DUMP_CMD", "matmul_w4a16/ins")
    os.makedirs("matmul_w4a16", exist_ok=True)

    torch.ops.my_ops.matmul_gptq_forward(
        x, q_weight, bias, q_scale_zp, q_scale_zp, group_size, weight_bits, output
    )
    return output


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, new_hidden_size, with_bias",
    [
        (1, 6, 3584, 4096, False),
        # (16, 1024, 7168, 8192, True),
        # (128, 4096, 8192, 8192, True),
    ],
)
def test_LLM_matmul_w4a16(
    batch_size,
    seq_len,
    hidden_size,
    new_hidden_size,
    with_bias,
    device,
    setup_random_seed,
):
    """
    测试 Matmul 的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        hidden_size: 隐藏层维度
        with_bias: 是否使用偏置参数
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
    """
    # 先默认一些通用参数
    dtype = torch.float16
    q_group_size = 128
    weight_bits = 4

    # 第一步：生成测试输入数据
    x = torch.randn((batch_size * seq_len, hidden_size), requires_grad=False)
    x_tpu = x.to(dtype).to(device)

    # 第二步：准备权重、缩放和偏置参数
    q_weight = None
    q_scale_zp = None
    bias = None
    if True:
        # 创建量化权重和缩放参数
        q_weight = torch.ones((new_hidden_size, hidden_size // 2), dtype=torch.uint8)
        q_scale_zp = torch.ones(
            (
                new_hidden_size,
                hidden_size // q_group_size * 2 + hidden_size // q_group_size // 2,
            ),
            dtype=torch.uint8,
        )

        tpu_q_weight = q_weight.to(device)
        tpu_q_scale_zp = q_scale_zp.to(device)

    if with_bias:
        # 创建偏置参数并转换到相应设备
        bias = torch.rand(new_hidden_size)
        tpu_bias = bias.to(dtype).to(device)

    # 第四步：运行 TPU 实现，在这里进行性能分析
    # 在 profiler 上下文中执行 TPU matmul w4a16 计算
    matmal_w4a16_tpu(
        x_tpu,
        tpu_q_weight,
        tpu_q_scale_zp,
        tpu_bias if with_bias else None,
        q_group_size,
        weight_bits,
    )

    # out_tpu = out_tpu.cpu()

    print_graph_summary()

    # 第五步：计算差异并验证结果
