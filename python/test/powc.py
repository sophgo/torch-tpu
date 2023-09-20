from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


class TestPow(nn.Module):
    def forward(self, x):
        return [x**2, x**3.5]
        # return [torch.ops.aten.abs(x), torch.ops.prims.abs(x), torch.abs(x)]


def case1():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(5),
        NumberFunc.gauss(),
    )

    # res = torch.rand(10, 10).to(device)[:, 1:].cpu()
    # res = torch.ones(35, 65535).to(device).int()
    # print(res)

    Evaluator().add_abs_evalute().evavlute([TestPow()], ipts)


if __name__ == "__main__":
    case1()
