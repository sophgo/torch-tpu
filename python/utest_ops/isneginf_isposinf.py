import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_bacis_info


def case1():
    # step0:set seed
    seed = 1000
    set_bacis_info(seed)

    input_data = [
        [torch.tensor([float('inf'), float('-inf'), 0.0, 1.0, -1.0, 100.0, -100.0])],
        [torch.tensor([float('-inf'), float('inf'), 0.0])],
        [torch.rand((10, 20)) - 0.5],  # normal random values
        [torch.tensor([float('-inf')] * 5 + [float('inf')] * 5 + [0.0] * 5)],  # mixed inf values
    ]
    device = "tpu:0"
    for input in input_data:
        input = input[0]
        isneg_cpu = torch.isneginf(input)
        isneg_tpu = torch.isneginf(input.to(device))
        ispos_cpu = torch.isposinf(input)
        ispos_tpu = torch.isposinf(input.to(device))

        status_isneginf = torch.equal(isneg_cpu, isneg_tpu.cpu())
        status_isposinf = torch.equal(ispos_cpu, ispos_tpu.cpu())

        res = status_isneginf and status_isposinf
        if not res:
            import sys
            sys.exit(255)
        else:
            print("isneginf and isposinf passed")
if __name__ == "__main__":
    case1()

