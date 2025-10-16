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

def case2():
    x = torch.randn((256,1,1,256), dtype=torch.float32)

    cpu_out = torch.sigmoid(x)
    tpu_out = torch.ops.aten.sigmoid(x.to(device))
    print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
    print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")

if __name__ == "__main__":
    case1()
    # case2()
