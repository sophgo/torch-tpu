import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
from torch_tpu.utils.compare import cos_sim, cal_diff

# 移除硬编码的随机种子和打印选项设置，这些现在由 conftest.py 中的 fixture 处理
# 移除硬编码的设备设置

@pytest.mark.parametrize(
    "input_shape",
    [
        (5, 5),
        (10, 10),
        (3, 20, 20),
        (3, 256),
    ],
)
def test_cumsum(input_shape, device, setup_random_seed):

    input_origin = torch.rand(input_shape).to(torch.bfloat16)

    input_cpu = F.softmax(input_origin, dim=-1)

    input_tpu = input_cpu.to(device)

    output_cpu = torch.ops.aten.cumsum(input_cpu, dim=-1).float().flatten()

    output_tpu = torch.cumsum(input_tpu, -1).to("cpu").float().flatten()

    csim = cos_sim(output_cpu.numpy(), output_tpu.numpy())
    cos_diff, RMSE, amax_diff = cal_diff(output_cpu, output_tpu, "cumsum")

    assert csim > 0.99 and cos_diff < 1e-4
