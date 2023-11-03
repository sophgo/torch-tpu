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

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class TestArcSinH(nn.Module):
    def forward(self, x):
        return [torch.ops.prims.asinh(x), torch.ops.aten.asinh(x)]


class TestArcCosH(nn.Module):
    def forward(self, x):
        return [torch.ops.prims.acosh(x), torch.ops.aten.acosh(x)]


class TestArcTanH(nn.Module):
    def forward(self, x):
        return [torch.ops.prims.atanh(x), torch.ops.aten.atanh(x)]


def test_case1():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([-10, 10]),
    )

    evaluate([TestArcSinH()], ipts)


def test_case2():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([1.1, 20]),  # > 1
    )

    evaluate([TestArcCosH()], ipts)


def test_case3():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([-0.9, 0.9]),  # TODO：atanh 接近无穷时精度会有问题
    )

    evaluate([TestArcTanH()], ipts)


if __name__ == "__main__":
    test_case1()
    test_case2()
    test_case3()
