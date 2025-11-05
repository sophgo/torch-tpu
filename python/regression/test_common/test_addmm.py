import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
from torch_tpu.utils.compare import cos_sim, cal_diff

@pytest.mark.parametrize(
    "m,k,n",
    [
        (2, 3, 3),
        (8, 2048, 1000),    # 8-64B ResNet-50/101/152
        (16, 2048, 1000),
        (32, 2048, 1000),
        (64, 2048, 1000),
    ],
)
def test_addmm(m, k, n, device, setup_random_seed):
    """torch.addmm"""
    mat = torch.rand(n).to(torch.bfloat16)
    mat1 = torch.rand(m, k).to(torch.bfloat16)
    mat2 = torch.rand(k, n).to(torch.bfloat16)

    # input_cpu = (mat, mat1, mat2)
    output_cpu = torch.addmm(mat, mat1, mat2).float().flatten()
    mat_tpu = mat.to(device)
    mat1_tpu = mat1.to(device)
    mat2_tpu = mat2.to(device)

    output_tpu = torch.addmm(mat_tpu, mat1_tpu, mat2_tpu).to("cpu").float().flatten()

    csim = cos_sim(output_cpu.numpy(), output_tpu.numpy())
    cos_diff, RMSE, amax_diff = cal_diff(output_cpu, output_tpu, "addmm")
    assert csim > 0.99 and cos_diff < 1e-4