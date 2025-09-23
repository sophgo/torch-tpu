from torch_tpu.utils.reflection import TensorLike
from torch import nn
import torch

import torch
from torch_tpu.utils.reflection.torch_inject import TensorLike


def is_tensor_like(a):
    assert isinstance(a, TensorLike)


def test_magic():
    a = torch.randint(1, 10, (10, 10), dtype=torch.int32)
    # add
    is_tensor_like(a + 1)
    is_tensor_like(1 + a)
    is_tensor_like(a + a)
    # sub
    is_tensor_like(a - a)
    is_tensor_like(a - 1)
    is_tensor_like(1 - a)
    # mul
    is_tensor_like(a * 2)
    is_tensor_like(2 * a)
    is_tensor_like(a * a)
    # div
    is_tensor_like(a / 2)
    is_tensor_like(2 / a)
    is_tensor_like(a / 2)
    # mod
    is_tensor_like(2 % a)
    is_tensor_like(a % 2)
    is_tensor_like(a % a)
    # pow
    is_tensor_like(a**2)
    is_tensor_like(2**a)
    is_tensor_like(a**a)

    # floor div
    is_tensor_like(a // 2)
    is_tensor_like(2 // a)
    is_tensor_like(a // a)

    # and
    is_tensor_like(a & 2)
    is_tensor_like(2 & a)
    is_tensor_like(a & a)

    # or
    is_tensor_like(a | 2)
    is_tensor_like(2 | a)
    is_tensor_like(a | a)

    # xor
    is_tensor_like(a ^ 2)
    is_tensor_like(2 ^ a)
    is_tensor_like(a ^ a)

    # lshift
    is_tensor_like(a << 2)
    is_tensor_like(2 << a)
    is_tensor_like(a << a)

    # rshift
    is_tensor_like(a >> 2)
    is_tensor_like(2 >> a)
    is_tensor_like(a >> a)

    # eq
    is_tensor_like(a == 2)
    is_tensor_like(2 == a)
    is_tensor_like(a == a)
    # ne
    is_tensor_like(a != 2)
    is_tensor_like(2 != a)
    is_tensor_like(a != a)
    # gt
    is_tensor_like(a > 2)
    is_tensor_like(2 > a)
    is_tensor_like(a > a)
    # ge
    is_tensor_like(a >= 2)
    is_tensor_like(2 >= a)
    is_tensor_like(a >= a)
    # lt
    is_tensor_like(a < 2)
    is_tensor_like(2 < a)
    is_tensor_like(a < a)
    # le
    is_tensor_like(a <= 2)
    is_tensor_like(2 <= a)
    is_tensor_like(a <= a)

    # matmul
    is_tensor_like(a @ a)


def test_torch_attr():
    a = torch.randn(10)
    assert isinstance(a, TensorLike)


def test_torch_parameter():
    a = nn.Parameter(torch.randn(10))

    assert isinstance(a, TensorLike)
    assert isinstance(a.data, TensorLike)
    assert isinstance(a._tensor, nn.Parameter)
