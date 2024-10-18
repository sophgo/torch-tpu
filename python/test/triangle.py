"""
测试类型：
F32,F16,BF16,INT8
shape: *args
domain: [left, right]
"""
import numpy as np
import torch
import torch.nn as nn
from test_utils import *
import torch.nn.functional as F
from tqdm import tqdm

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


class TestSin(nn.Module):
    def forward(self, x):
        return [torch.ops.prims.sin(x), torch.ops.aten.sin(x)]


class TestCos(nn.Module):
    def forward(self, x):
        return [torch.ops.prims.cos(x), torch.ops.aten.cos(x)]


class TestTan(nn.Module):
    def forward(self, x):
        return [torch.ops.prims.tan(x), torch.ops.aten.tan(x)]


def test_case1():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([-torch.pi * 3, torch.pi * 3]),
    )

    evaluate([TestSin(), TestCos()], ipts)


def test_case2():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace(
            [-torch.pi / 2 + 0.4, torch.pi / 2 - 0.4]
        ),  # TODO：tan 接近无穷时精度会有问题
    )

    evaluate([TestTan()], ipts)


if __name__ == "__main__":
    test_case1()
    test_case2()
