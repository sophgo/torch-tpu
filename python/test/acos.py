import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import *

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class TestACos(nn.Module):
    def forward(self, x):
        return [torch.ops.aten.acos(x), torch.ops.prims.acos(x)]


def case1():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([-0.01, -0.99]),  # acos input domain [-1, 1]
    )

    Evaluator().add_abs_evalute(f32_eps=1e-5).evavlute([TestACos()], ipts)


def case2():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(),
        NumberFunc.linespace([0.01, 0.99]),  # acos input domain [-1, 1]
    )

    Evaluator().add_abs_evalute(f32_eps=1e-5).evavlute([TestACos()], ipts)


if __name__ == "__main__":
    case1()
    case2()
