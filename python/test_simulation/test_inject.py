from torch_tpu.utils.reflection import TensorLike
from torch import nn
import torch


def test_torch_attr():
    a = torch.randn(10)
    assert isinstance(a, TensorLike)


def test_torch_parameter():
    a = nn.Parameter(torch.randn(10))

    assert isinstance(a, TensorLike)
    assert isinstance(a.data, TensorLike)
    assert isinstance(a._tensor, nn.Parameter)
