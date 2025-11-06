import pytest
import torch
import torch.nn as nn
import copy
import torch_tpu
from torch_tpu.utils.compare import cos_sim, cal_diff

@pytest.mark.parametrize(
    "B,H,W,IC,OC,K,stride,pad,dilation,enable_bias,enable_gradinput,enable_gradweight",
    [
        (8, 224, 224, 3, 64, 3, 1, 1,1, 0, 1, 1),
        (8, 64, 64, 3, 32, 3, 1, 1, 1, 0, 1, 1),
        (8, 40, 40, 256, 256, 3, 2, 1,1, 0, 1, 1),
        (8, 80, 80, 128, 256, 3, 2, 1, 1, 0, 1, 1),
        (8, 640, 640, 3, 32, 6, 2, 2, 1, 0, 1, 1),
        (8, 56, 56, 64, 64, 3, 1, 1, 1, 0, 1, 1),
        (8, 56, 56, 256, 64, 1, 1, 0, 1, 0, 1, 1),
        (8, 20, 20, 512, 1024, 1, 1, 0, 1, 0, 1, 1),
        (8, 20, 20, 1024, 512, 1, 1, 0, 1, 0, 1, 1),
        (8, 40, 40, 512, 128, 1, 1, 0, 1, 0, 1, 1),
        (8, 20, 20, 512, 255, 1, 1, 0, 1, 1, 1, 1),
        (8, 40, 40, 256, 255, 1, 1, 0, 1, 1, 1, 1),
        (8, 80, 80, 128, 255, 1, 1, 0, 1, 1, 1, 1),
        (64, 224, 224, 3, 64, 3, 1, 1, 1, 0, 1, 1),
        (64, 640, 640, 3, 32, 3, 2, 2, 1, 0, 1, 1),
    ],
)
def test_convolution_backward(B, H, W, IC, OC, K, stride, pad, dilation,enable_bias, enable_gradinput, enable_gradweight, device, setup_random_seed):
    kh, kw = K, K
    ih, iw = H, W
    strides = [stride, stride]
    padding = [pad, pad]
    dilations = [dilation, dilation]

    oh = (ih + 2*padding[0] - (dilations[0]*(kh-1) + 1)) // strides[0] + 1
    ow = (iw + 2*padding[1] - (dilations[1]*(kw-1) + 1)) // strides[1] + 1

    grad_out_cpu = torch.randn(B, OC, oh, ow, dtype=torch.bfloat16)
    weight_cpu = torch.randn(OC, IC, kh, kw, dtype=torch.bfloat16)
    x_cpu = torch.randn(B, IC, ih, iw, dtype=torch.bfloat16)

    grad_out_tpu = copy.deepcopy(grad_out_cpu)
    grad_out_tpu = grad_out_tpu.to(device)
    weight_tpu = copy.deepcopy(weight_cpu)
    weight_tpu = weight_tpu.to(device)
    x_tpu = copy.deepcopy(x_cpu)
    x_tpu = x_tpu.to(device)

    output_mask = [enable_gradinput, enable_gradweight, enable_bias]

    grad_input_cpu, grad_weight_cpu, grad_bias_cpu = torch.ops.aten.convolution_backward(
        grad_out_cpu.float(), x_cpu.float(), weight_cpu.float(),
        None, strides, padding, dilations,
        False, [0], 1, output_mask
    )

    grad_input_tpu, grad_weight_tpu, grad_bias_tpu = torch.ops.aten.convolution_backward(
        grad_out_tpu.float(), x_tpu.float(), weight_tpu.float(),
        None, strides, padding, dilations,
        False, [0], 1, output_mask
    )

    res_tpu = []
    res_cpu = []

    if enable_gradinput:
        res_tpu.append(grad_input_tpu)
        res_cpu.append(grad_input_cpu)

    if enable_gradweight:
        res_tpu.append(grad_weight_tpu)
        res_cpu.append(grad_weight_cpu)

    if enable_bias:
        res_tpu.append(grad_bias_tpu)
        res_cpu.append(grad_bias_cpu)

    for i in range(len(res_cpu)):
        csim = cos_sim(res_tpu[i].to("cpu").detach().numpy(), res_cpu[i].detach().numpy())
        cos_diff, RMSE, amax_diff = cal_diff(res_tpu[i].to("cpu"), res_cpu[i], "convolution_backward")
        #print(csim, cos_diff)
        assert csim > 0.99 and cos_diff < 1e-3