import torch
from test_utils import *
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class TestLog(nn.Module):
    def forward(self, x: torch.Tensor):
        if x.device.type == "cpu":
            x = x.float()
        return x.log()


class TestLog2(nn.Module):
    def forward(self, x: torch.Tensor):
        return [x.log2()]


class TestLog10(nn.Module):
    def forward(self, x: torch.Tensor):
        return [x.log10()]


class TestLog1p(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.log1p()


def case1():
    ipts = InputIter.create(
        DTypeIter.float_type(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([0.2, 100]),
    )

    Evaluator().add_abs_evalute().evavlute([TestLog()], ipts)


def case2():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([0.2, 100]),
    )
    Evaluator().add_abs_evalute().evavlute([TestLog2(), TestLog10(), TestLog1p()], ipts)


if __name__ == "__main__":
    case1()
    case2()
