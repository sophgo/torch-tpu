from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


class TestAbs(nn.Module):
    def forward(self, x):
        return [x.mean()]


def case1():
    ipts = InputIter.create(
        DTypeIter.float32(),
        ShapeIter.any_shape(5),
        NumberFunc.linespace([-100, 100]),
    )
    Evaluator().add_abs_evalute().evavlute([TestAbs()], [torch.tensor(1.2)])
    Evaluator().add_abs_evalute(1e-5).evavlute([TestAbs()], ipts)


if __name__ == "__main__":
    case1()
