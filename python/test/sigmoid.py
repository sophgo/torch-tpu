from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


class TestSigmoid(nn.Module):
    def forward(self, x):
        return [torch.ops.aten.sigmoid(x), torch.sigmoid(x)]


def case1():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(5),
        NumberFunc.linespace([-100, 100]),
    )
    Evaluator().add_abs_evalute().evavlute([TestSigmoid()], ipts)


if __name__ == "__main__":
    case1()