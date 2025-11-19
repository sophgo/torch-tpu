import pytest
import torch
import torch_tpu
import torch.nn as nn
import copy
from torch_tpu.utils.compare import cos_sim, cal_diff
import torch.nn.functional as F

@pytest.mark.parametrize(
    "shape",
    [
        (1128, 20),
    ],
)
def binary_cross_entropy_with_logits(shape):
    device = "tpu:0"
    input       = torch.randn(shape)
    target      = torch.empty(shape).random_(2)
    input_tpu   = copy.deepcopy(input).to(device)
    target_tpu  =  copy.deepcopy(target).to(device)
    input.requires_grad     = True
    input_tpu.requires_grad = True

    loss        = F.binary_cross_entropy_with_logits(input, target)
    loss.backward()

    loss_tpu    = F.binary_cross_entropy_with_logits(input_tpu, target_tpu)
    loss_tpu.backward()

    loss_diff       = loss - loss_tpu.cpu()
    input_grad_diff = input.grad - input_tpu.grad.cpu()
    assert torch.max(abs(loss_diff)) < 1e-4
    assert torch.max(abs(input_grad_diff)) < 1e-4


if __name__ == "__main__":
    binary_cross_entropy_with_logits((1128, 20))