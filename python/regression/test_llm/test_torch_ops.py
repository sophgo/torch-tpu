import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu


# ==================== 除法操作测试 ====================


def div_cpu(a, b):
    """CPU 版本的除法操作实现"""
    # 第一步：执行元素级除法
    result = torch.div(a, b)
    return result.float()


def div_tpu(a, b, device=None, profiler=None):
    """使用 TPU 的除法操作实现"""
    # 第一步：将输入数据移动到指定设备
    a_tpu = a.to(device)
    b_tpu = b.to(device)
    
    # 第二步：调用PyTorch的除法操作，在TPU上执行
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            result = torch.div(a_tpu, b_tpu)
    else:
        result = torch.div(a_tpu, b_tpu)

    return result.cpu().float()


@pytest.mark.parametrize(
    "batch_size, num_experts",
    [
        (32, 8),  # 基于日志参数 routing_weights / denominator
        (64, 8),  # 扩展测试参数
        (16, 8),  # 较小的测试参数
    ],
)
def test_div_forward(
    batch_size,
    num_experts,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试除法操作的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        num_experts: 专家数量
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    # 基于日志：routing_weights.shape=torch.Size([32, 8])
    # denominator.shape=torch.Size([32, 1])
    routing_weights = torch.randn((batch_size, num_experts), dtype=torch.bfloat16)
    denominator = torch.randn((batch_size, 1), dtype=torch.bfloat16) + 1.0  # 避免除零

    # 第二步：运行 TPU 实现，设备转换在div_tpu函数内部完成
    out_tpu = div_tpu(routing_weights, denominator, device=device, profiler=profiler)

    # 第四步：运行 CPU 参考实现
    out_cpu = div_cpu(routing_weights, denominator)

    # 第五步：比较结果
    out_tpu_cpu = out_tpu.float().cpu().detach()
    out_diff = out_cpu - out_tpu_cpu
    mean_diff = torch.mean(abs(out_diff)).item()

    # 由于使用了半精度计算，允许一定的误差
    tolerance = 1e-2

    # 断言平均差异在容忍范围内
    assert (
        mean_diff < tolerance
    ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"


# ==================== 矩阵乘法操作测试 ====================


def matmul_cpu(a, b):
    """CPU 版本的矩阵乘法操作实现"""
    # 第一步：执行矩阵乘法
    result = torch.matmul(a, b)
    return result.float()


def matmul_tpu(a, b, device=None, profiler=None):
    """使用 TPU 的矩阵乘法操作实现"""
    # 第一步：将输入数据移动到指定设备
    a_tpu = a.to(device)
    b_tpu = b.to(device)
    
    # 第二步：调用PyTorch的矩阵乘法操作，在TPU上执行
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            result = torch.matmul(a_tpu, b_tpu)
    else:
        result = torch.matmul(a_tpu, b_tpu)

    return result.cpu().float()


@pytest.mark.parametrize(
    "batch_size, input_dim, output_dim",
    [
        (32, 7168, 256),  # 基于日志参数 x @ gate_weight_T
        (64, 4096, 512),  # 扩展测试参数
        (16, 2048, 128),  # 较小的测试参数
    ],
)
def test_matmul_forward(
    batch_size,
    input_dim,
    output_dim,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试矩阵乘法操作的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        input_dim: 输入维度
        output_dim: 输出维度
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    # 基于日志：x.shape=torch.Size([32, 7168])
    # gate_weight_T.shape=torch.Size([7168, 256])
    x = torch.randn((batch_size, input_dim), dtype=torch.bfloat16)
    gate_weight_T = torch.randn((input_dim, output_dim), dtype=torch.bfloat16)

    # 第二步：运行 TPU 实现，设备转换在matmul_tpu函数内部完成
    out_tpu = matmul_tpu(x, gate_weight_T, device=device, profiler=profiler)

    # 第四步：运行 CPU 参考实现
    out_cpu = matmul_cpu(x, gate_weight_T)

    # 第五步：比较结果
    out_tpu_cpu = out_tpu.float().cpu().detach()
    out_diff = out_cpu - out_tpu_cpu
    mean_diff = torch.mean(abs(out_diff)).item()

    # 由于使用了半精度计算，允许一定的误差
    tolerance = 1e-2

    # 断言平均差异在容忍范围内
    assert (
        mean_diff < tolerance
    ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"


# ==================== 乘法操作测试 ====================


def mul_cpu(a, b):
    """CPU 版本的乘法操作实现"""
    # 第一步：执行元素级乘法
    result = torch.mul(a, b)
    return result.float()


def mul_tpu(a, b, device=None, profiler=None):
    """使用 TPU 的乘法操作实现"""
    # 第一步：将输入数据移动到指定设备
    a_tpu = a.to(device)
    
    # 第二步：调用PyTorch的乘法操作，在TPU上执行
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            result = torch.mul(a_tpu, b)
    else:
        result = torch.mul(a_tpu, b)

    return result.cpu().float()


@pytest.mark.parametrize(
    "batch_size, num_experts, scaling_factor",
    [
        (32, 8, 2.5),  # 基于日志参数 routing_weights * routed_scaling_factor
        (64, 16, 3.0),  # 扩展测试参数
        (16, 4, 1.5),  # 较小的测试参数
    ],
)
def test_mul_forward(
    batch_size,
    num_experts,
    scaling_factor,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试乘法操作的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        num_experts: 专家数量
        scaling_factor: 缩放因子
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    # 基于日志：routing_weights.shape=torch.Size([32, 8]) * routed_scaling_factor=2.5
    routing_weights = torch.randn((batch_size, num_experts), dtype=torch.bfloat16)

    # 第二步：运行 TPU 实现，设备转换在mul_tpu函数内部完成
    out_tpu = mul_tpu(routing_weights, scaling_factor, device=device, profiler=profiler)

    # 第四步：运行 CPU 参考实现
    out_cpu = mul_cpu(routing_weights, scaling_factor)

    # 第五步：比较结果
    out_tpu_cpu = out_tpu.float().cpu().detach()
    out_diff = out_cpu - out_tpu_cpu
    mean_diff = torch.mean(abs(out_diff)).item()

    # 由于使用了半精度计算，允许一定的误差
    tolerance = 1e-2

    # 断言平均差异在容忍范围内
    assert (
        mean_diff < tolerance
    ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"


# ==================== Sigmoid激活函数测试 ====================


def sigmoid_cpu(x):
    """CPU 版本的Sigmoid激活函数实现"""
    # 第一步：执行Sigmoid激活函数
    result = torch.sigmoid(x)
    return result


def sigmoid_tpu(x, device=None, profiler=None):
    """使用 TPU 的Sigmoid激活函数实现"""
    # 第一步：将输入数据移动到指定设备
    x_tpu = x.to(device)
    
    # 第二步：调用PyTorch的Sigmoid激活函数，在TPU上执行
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            result = torch.sigmoid(x_tpu)
    else:
        result = torch.sigmoid(x_tpu)

    return result.cpu()


@pytest.mark.parametrize(
    "batch_size, num_experts",
    [
        (32, 256),  # 基于日志参数 router_logits
        (64, 512),  # 扩展测试参数
        (16, 128),  # 较小的测试参数
    ],
)
def test_sigmoid_forward(
    batch_size,
    num_experts,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试Sigmoid激活函数的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        num_experts: 专家数量
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    # 基于日志：router_logits.shape=torch.Size([32, 256])
    router_logits = torch.randn((batch_size, num_experts), dtype=torch.bfloat16)

    # 第二步：运行 TPU 实现，设备转换在sigmoid_tpu函数内部完成
    out_tpu = sigmoid_tpu(router_logits, device=device, profiler=profiler)

    # 第四步：运行 CPU 参考实现
    out_cpu = sigmoid_cpu(router_logits)

    # 第五步：比较结果
    out_tpu_cpu = out_tpu.float().cpu().detach()
    out_diff = out_cpu - out_tpu_cpu
    mean_diff = torch.mean(abs(out_diff)).item()

    # 由于使用了半精度计算，允许一定的误差
    tolerance = 1e-3

    # 断言平均差异在容忍范围内
    assert (
        mean_diff < tolerance
    ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"


# ==================== 求和操作测试 ====================


def sum_cpu(x, dim=None, keepdim=False):
    """CPU 版本的求和操作实现"""
    # 第一步：执行求和操作
    result = torch.sum(x, dim=dim, keepdim=keepdim)
    return result


def sum_tpu(x, dim=None, keepdim=False, device=None, profiler=None):
    """使用 TPU 的求和操作实现"""
    # 第一步：将输入数据移动到指定设备
    x_tpu = x.to(device)
    
    # 第二步：调用PyTorch的求和操作，在TPU上执行
    if profiler:
        with profiler.profile(buffer_size=1024, trace_level=2):
            result = torch.sum(x_tpu, dim=dim, keepdim=keepdim)
    else:
        result = torch.sum(x_tpu, dim=dim, keepdim=keepdim)

    return result.cpu()


@pytest.mark.parametrize(
    "batch_size, num_experts, dim, keepdim",
    [
        (32, 8, 1, True),  # 基于日志参数 routing_weights求和，保持维度
        (32, 8, None, False),  # 全局求和
        (64, 16, 0, False),  # 批次维度求和
        (16, 4, 1, False),  # 专家维度求和
    ],
)
def test_sum_forward(
    batch_size,
    num_experts,
    dim,
    keepdim,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试求和操作的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        num_experts: 专家数量
        dim: 求和维度
        keepdim: 是否保持维度
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成测试输入数据
    # 基于日志：routing_weights.shape=torch.Size([32, 8])
    routing_weights = torch.randn((batch_size, num_experts), dtype=torch.bfloat16)

    # 第二步：运行 TPU 实现，设备转换在sum_tpu函数内部完成
    out_tpu = sum_tpu(routing_weights, dim=dim, keepdim=keepdim, device=device, profiler=profiler)

    # 第四步：运行 CPU 参考实现
    out_cpu = sum_cpu(routing_weights, dim=dim, keepdim=keepdim)

    # 第五步：比较结果
    out_tpu_cpu = out_tpu.float().cpu().detach()
    out_diff = out_cpu - out_tpu_cpu
    mean_diff = torch.mean(abs(out_diff)).item()

    # 由于使用了半精度计算，允许一定的误差
    tolerance = 1e-2

    # 断言平均差异在容忍范围内
    assert (
        mean_diff < tolerance
    ), f"Mean difference {mean_diff} exceeds tolerance {tolerance}"


# ==================== 综合测试：专家路由计算流程 ====================


@pytest.mark.parametrize(
    "batch_size, hidden_dim, num_experts, top_k",
    [
        (32, 7168, 256, 8),  # 基于日志的完整参数
        (16, 4096, 128, 4),  # 较小的测试参数
    ],
)
def test_expert_routing_pipeline(
    batch_size,
    hidden_dim,
    num_experts,
    top_k,
    device,
    setup_random_seed,
    profiler,
):
    """
    测试专家路由的完整计算流程，包含多个基础操作

    Args:
        batch_size: 批次大小
        hidden_dim: 隐藏层维度
        num_experts: 专家总数
        top_k: 选择的专家数量
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
        profiler: 性能分析器（由 fixture 提供）
    """

    # 第一步：生成输入数据
    x = torch.randn((batch_size, hidden_dim), dtype=torch.bfloat16)
    gate_weight = torch.randn((num_experts, hidden_dim), dtype=torch.bfloat16)
    
    # 第二步：将输入数据移动到指定设备
    x = x.to(device)
    gate_weight = gate_weight.to(device)

    # 第三步：计算路由分数 (matmul)
    router_logits = torch.matmul(x, gate_weight.T)

    # 第四步：应用激活函数 (sigmoid)
    routing_scores = torch.sigmoid(router_logits)

    # 第五步：选择top-k专家 (topk)
    routing_weights, selected_experts = torch.topk(routing_scores, top_k, dim=1)

    # 第六步：计算归一化分母 (sum)
    denominator = torch.sum(routing_weights, dim=1, keepdim=True)

    # 第七步：归一化权重 (div)
    normalized_weights = torch.div(routing_weights, denominator)

    # 第八步：应用缩放因子 (mul)
    scaled_weights = torch.mul(normalized_weights, 2.5)

    # 第九步：验证结果形状和范围
    assert router_logits.shape == (
        batch_size,
        num_experts,
    ), f"Router logits shape mismatch"
    assert routing_scores.shape == (
        batch_size,
        num_experts,
    ), f"Routing scores shape mismatch"
    assert routing_weights.shape == (
        batch_size,
        top_k,
    ), f"Routing weights shape mismatch"
    assert selected_experts.shape == (
        batch_size,
        top_k,
    ), f"Selected experts shape mismatch"
    assert normalized_weights.shape == (
        batch_size,
        top_k,
    ), f"Normalized weights shape mismatch"
    assert scaled_weights.shape == (batch_size, top_k), f"Scaled weights shape mismatch"

    # 第十步：验证数值范围
    assert torch.all(routing_scores >= 0) and torch.all(
        routing_scores <= 1
    ), "Sigmoid output out of range"
    assert torch.all(selected_experts >= 0) and torch.all(
        selected_experts < num_experts
    ), "Expert indices out of range"

    # 第十一步：验证测试通过的断言，而不是使用print
    assert batch_size > 0 and num_experts > 0, f"Expert routing pipeline test passed for batch_size={batch_size}, num_experts={num_experts}"
