from typing import Any
from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


class TestClone:
    def __call__(self, x) -> Any:
        return self.forward(x)

    def forward(self, x):
        return torch.ops.aten.clone(x)


def case1():
    # test dtype

    for dtype in DTypeIter.all():
        for shape in ShapeIter.any_shape():
            cpu_data = torch.ones(*shape, dtype=dtype)
            Evaluator().add_abs_evalute().evavlute([TestClone()], [cpu_data])


if __name__ == "__main__":
    case1()
