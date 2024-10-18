import torch
from test_utils import *
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


def case1():
    for typ in DTypeIter.float_type() + [torch.int32]:
        for size in ShapeIter.any_shape():
            for scalar in [1.0, -1.0, 5.0]:
                cpu = torch.full(size, scalar, dtype=typ)
                tpu = torch.full(size, scalar, dtype=typ, device=device)
                assert (cpu == tpu.cpu()).all()


if __name__ == "__main__":
    case1()
