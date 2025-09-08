import torch
import torch.nn.functional as F
import pytest
import torch_tpu
import numpy as np

# 移除硬编码的随机种子和设备设置，这些现在由 conftest.py 中的 fixture 处理


def gt_argmax(logits):
    """Ground truth 实现，用于对比验证"""
    # 第一步：在最后一个维度上计算argmax，保持维度
    argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
    return argmax_token_ids


def argmax_v1(logits, device):
    """使用标准PyTorch实现的版本"""
    # 第一步：将数据移动到指定设备
    logits = logits.clone().to(device)
    # 第二步：计算argmax
    argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

    return argmax_token_ids.cpu()


def cmp_argmax(argmax_gt, argmax_test):
    """比较两个argmax结果是否一致"""
    # 第一步：验证形状是否一致
    assert (
        argmax_gt.shape == argmax_test.shape
    ), f"Shape mismatch: {argmax_gt.shape} vs {argmax_test.shape}"

    # 第二步：验证数值是否完全一致
    assert torch.equal(argmax_gt, argmax_test), "Argmax results do not match"


@pytest.mark.parametrize(
    "batch_size,vocab_size,dtype",
    [
        (1, 129280, torch.float32),
        (9, 129280, torch.float32),
        (16, 129280, torch.float32),
        (32, 129280, torch.float32),
        (1, 129280, torch.bfloat16),
        (9, 129280, torch.bfloat16),
        (16, 129280, torch.bfloat16),
        (32, 129280, torch.bfloat16),
    ],
)
def test_argmax_precision(batch_size, vocab_size, dtype, device):
    """测试不同精度下的 argmax 计算"""

    # 第一步：生成测试数据
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)

    # 第二步：转换到指定精度
    logits = logits.to(dtype)

    # 第三步：运行ground truth版本（使用float32作为参考）
    argmax_gt = gt_argmax(logits.float())

    # 第四步：运行测试版本
    argmax_test = argmax_v1(logits.clone(), device)

    # 第五步：验证结果的正确性
    cmp_argmax(argmax_gt, argmax_test)

