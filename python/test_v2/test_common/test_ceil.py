import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu

# 移除硬编码的随机种子和打印选项设置，这些现在由 conftest.py 中的 fixture 处理
# 移除硬编码的设备设置


@pytest.mark.parametrize(
    "input_shape",
    [
        (5, 5),
        (10, 10),
        (3, 20, 20),
        (3, 20, 20, 20),
    ],
)
def test_ceil(input_shape, device, setup_random_seed):

    input_origin = torch.rand(input_shape) * 2000 - 1000

    input_tpu = input_origin.to(device)

    output_cpu_prims = torch.ops.prims.ceil(input_origin)

    output_tpu_prims = torch.ops.prims.ceil(input_tpu).cpu()
    assert torch.allclose(output_cpu_prims, output_tpu_prims)
