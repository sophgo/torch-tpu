from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class TestAbs(nn.Module):
    def forward(self, x):
        return [torch.ops.aten.abs(x), torch.ops.prims.abs(x), torch.abs(x)]


def case1():
    ipts = InputIter.create(
        DTypeIter.float_type(),
        ShapeIter.any_shape(5),
        NumberFunc.linespace([-100, 100]),
    )
    Evaluator().add_abs_evalute().evavlute([TestAbs()], ipts)


if __name__ == "__main__":
    case1()
