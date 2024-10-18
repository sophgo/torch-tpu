from typing import Any
from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


convert = [
    [torch.float32, torch.int8],
    [torch.float32, torch.float16],
    [torch.float32, torch.bfloat16],
    [torch.float16, torch.float32],
    # [torch.float16, torch.bfloat16],
    [torch.float16, torch.int8],
    [torch.bfloat16, torch.float32],
    # [torch.bfloat16, torch.float16], # can not
    [torch.bfloat16, torch.int8],
    [torch.int8, torch.float32],
    [torch.int8, torch.float16],
    [torch.int8, torch.bfloat16],
]


class TestConvert:
    def __init__(self, to_dtype) -> None:
        self.to_dtype = to_dtype

    def __call__(self, x) -> Any:
        return self.forward(x)

    def forward(self, x):
        return torch.ops.aten._to_copy(x, dtype=self.to_dtype)


def case1():
    # test dtype

    for a, b in convert:
        for shape in ShapeIter.any_shape():
            cpu_data = torch.ones(*shape, dtype=a)
            Evaluator().add_abs_evalute().evavlute([TestConvert(b)], [cpu_data])


if __name__ == "__main__":
    case1()
