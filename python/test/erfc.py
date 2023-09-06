from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class TestErfC(nn.Module):
    def forward(self, x):
        return [torch.ops.aten.erfc(x), torch.ops.prims.erfc(x), torch.erfc(x)]


def case1():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(5),
        NumberFunc.linespace([-100, 100]),
    )
    Evaluator().add_abs_evalute().evavlute([TestErfC()], ipts)


if __name__ == "__main__":
    case1()
