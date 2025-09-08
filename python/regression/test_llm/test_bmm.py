import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu


def bmm_cpu(a, b):
    """CPU 版本的批量矩阵乘法实现"""
    # 第一步：执行批量矩阵乘法
    result = torch.bmm(a, b)
    return result


def bmm_tpu(a, b, device=None, profiler=None):
    """使用 TPU 的批量矩阵乘法实现"""
    # 第一步：将输入数据移动到指定设备
    a_tpu = a.to(device)
    b_tpu = b.to(device)
    
    # 第二步：调用PyTorch的批量矩阵乘法，在TPU上执行
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            result = torch.bmm(a_tpu, b_tpu)
    else:
        result = torch.bmm(a_tpu, b_tpu)

    return result.cpu()


# @pytest.mark.parametrize(
#     "batch_size, dim1, dim2, dim3",
#     [
#         # (32, 7168, 8, 1),  # b32, maybe too large for regression
#         # (64, 4096, 16, 1),  # 扩展测试参数
#         # (16, 2048, 32, 2),  # 不同的批次和维度组合
#         (8, 1024, 64, 4),  # 较小的测试参数
#     ],
# )
# def test_bmm_forward(
#     batch_size,
#     dim1,
#     dim2,
#     dim3,
#     device,
#     setup_random_seed,
#     profiler,
# ):
#     """
#     测试批量矩阵乘法的 TPU 实现与 CPU 参考实现的一致性

#     Args:
#         batch_size: 批次大小
#         dim1: 第一个矩阵的第二维度
#         dim2: 矩阵乘法的共享维度
#         dim3: 第二个矩阵的第三维度
#         device: 测试设备（由 fixture 提供）
#         setup_random_seed: 随机种子设置（由 fixture 提供）
#         profiler: 性能分析器（由 fixture 提供）
#     """

#     # 第一步：生成测试输入数据
#     # 基于日志：transposed_experts.shape=torch.Size([32, 7168, 8])
#     # unsqueezed_weights.shape=torch.Size([32, 8, 1])
#     a = torch.randn((batch_size, dim1, dim2), dtype=torch.bfloat16)
#     b = torch.randn((batch_size, dim2, dim3), dtype=torch.bfloat16)

#     # 第二步：运行 TPU 实现，设备转换在bmm_tpu函数内部完成
#     out_tpu = bmm_tpu(a, b, device=device, profiler=profiler)

#     # 第三步：运行 CPU 参考实现
#     out_cpu = bmm_cpu(a, b)

#     # 第四步：比较结果
#     out_tpu_cpu = out_tpu.float().cpu().detach()
#     out_diff = out_cpu - out_tpu_cpu
#     mean_diff = torch.mean(abs(out_diff)).item()

#     # 由于使用了半精度计算，允许一定的误差
#     tolerance = 1e-2

#     # 断言平均差异在容忍范围内
#     assert (
#         mean_diff < tolerance
#     ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"


# @pytest.mark.parametrize(
#     "batch_size, experts_dim, num_experts",
#     [
#         (32, 7168, 8),  # 基于日志的专家混合参数
#         # (64, 4096, 16),  # 更多专家的配置
#         # (16, 2048, 4),   # 较小的专家配置
#     ],
# )
# def test_expert_routing_bmm(
#     batch_size,
#     experts_dim,
#     num_experts,
#     device,
#     setup_random_seed,
#     profiler,
# ):
#     """
#     测试专家路由中的批量矩阵乘法操作

#     Args:
#         batch_size: 批次大小
#         experts_dim: 专家模型的维度
#         num_experts: 专家数量
#         device: 测试设备（由 fixture 提供）
#         setup_random_seed: 随机种子设置（由 fixture 提供）
#         profiler: 性能分析器（由 fixture 提供）
#     """

#     # 第一步：生成专家输出（模拟多个专家的输出）
#     # 基于日志：transposed_experts.shape=torch.Size([32, 7168, 8])
#     transposed_experts = torch.randn(
#         (batch_size, experts_dim, num_experts), dtype=torch.bfloat16
#     )

#     # 第二步：生成路由权重（每个样本对专家的权重）
#     # 基于日志：unsqueezed_weights.shape=torch.Size([32, 8, 1])
#     routing_weights = torch.randn((batch_size, num_experts), dtype=torch.bfloat16)
#     unsqueezed_weights = routing_weights.unsqueeze(-1)  # 增加维度以适配bmm

#     # 第三步：运行 TPU 实现，设备转换在bmm_tpu函数内部完成
#     weighted_expert_output_tpu = bmm_tpu(
#         transposed_experts, unsqueezed_weights, device=device, profiler=profiler
#     )

#     # 第四步：运行 CPU 参考实现
#     weighted_expert_output_cpu = bmm_cpu(transposed_experts, unsqueezed_weights)

#     # 第五步：验证输出形状
#     expected_shape = (batch_size, experts_dim, 1)
#     assert (
#         weighted_expert_output_tpu.shape == expected_shape
#     ), f"TPU output shape mismatch: {weighted_expert_output_tpu.shape} vs {expected_shape}"
#     assert (
#         weighted_expert_output_cpu.shape == expected_shape
#     ), f"CPU output shape mismatch: {weighted_expert_output_cpu.shape} vs {expected_shape}"

#     # 第六步：比较结果
#     out_tpu_cpu = weighted_expert_output_tpu.float().cpu().detach()
#     out_diff = weighted_expert_output_cpu - out_tpu_cpu
#     mean_diff = torch.mean(abs(out_diff)).item()

#     # 由于使用了半精度计算，允许一定的误差
#     tolerance = 1e-2

#     # 断言平均差异在容忍范围内
#     assert (
#         mean_diff < tolerance
#     ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"
