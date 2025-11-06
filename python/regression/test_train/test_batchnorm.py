import pytest
import torch
import torch_tpu
import torch.nn as nn
import copy
from torch_tpu.utils.compare import cos_sim, cal_diff

@pytest.mark.parametrize(
    "shape",
    [
        (8, 64, 112, 112),
        (8, 32, 35, 35),
        (128, 128, 40, 40)
    ],
)
def test_batchnorm_train(shape, device, setup_random_seed):
    input_cpu = torch.rand((shape))
    input_tpu = copy.deepcopy(input_cpu)
    input_tpu = input_tpu.to(device)

    net_cpu = nn.BatchNorm2d(shape[1])
    net_tpu = copy.deepcopy(net_cpu)
    net_tpu = net_tpu.to(device)

    input_cpu.requires_grad = True
    input_tpu.requires_grad = True

    output_cpu = net_cpu(input_cpu)
    output_tpu = net_tpu(input_tpu)

    grad_output_cpu = torch.ones(output_cpu.shape) 
    grad_output_tpu = copy.deepcopy(grad_output_cpu)
    grad_output_tpu =  grad_output_tpu.to(device)

    output_cpu.backward(grad_output_cpu)
    output_tpu.backward(grad_output_tpu)
    
    res_tpu = [output_tpu, net_tpu.running_mean, net_tpu.running_var] #to do: add input_tpu.grad
    res_cpu = [output_cpu, net_cpu.running_mean, net_cpu.running_var] #to do: add input_cpu.grad

    for i in range(len(res_cpu)):
        csim = cos_sim(res_tpu[i].to("cpu").detach().numpy(), res_cpu[i].detach().numpy())
        cos_diff, RMSE, amax_diff = cal_diff(res_tpu[i].to("cpu"), res_cpu[i], "batchnorm_train")
        #print(csim, cos_diff)
        assert csim > 0.99 and cos_diff < 1e-4

@pytest.mark.parametrize(
    "shape",
    [
        (8, 32, 35, 35),
    ],
)
def test_batchnorm_forward(shape, device, setup_random_seed):
    input_cpu = torch.rand((shape))
    input_tpu = copy.deepcopy(input_cpu)
    input_tpu = input_tpu.to(device)

    net_cpu = nn.BatchNorm2d(shape[1])
    net_tpu = copy.deepcopy(net_cpu)
    net_tpu = net_tpu.to(device)

    input_cpu.requires_grad = True
    input_tpu.requires_grad = True

    output_cpu = net_cpu(input_cpu)
    output_tpu = net_tpu(input_tpu)
    
    res_tpu = [output_tpu]
    res_cpu = [output_cpu]

    for i in range(len(res_cpu)):
        csim = cos_sim(res_tpu[i].to("cpu").detach().numpy(), res_cpu[i].detach().numpy())
        cos_diff, RMSE, amax_diff = cal_diff(res_tpu[i].to("cpu"), res_cpu[i], "batchnorm_train")
        assert csim > 0.99 and cos_diff < 1e-4