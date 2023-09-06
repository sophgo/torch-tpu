import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def test_max_pool2d_with_indices():
    input_tensor = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    pooled_output, indices = max_pool2d_with_indices(input_tensor, kernel_size=2)
    input_tensor_tpu = input_tensor.clone().to(device)
    pooled_output_tpu, indices_tpu = max_pool2d_with_indices(input_tensor_tpu, kernel_size=2)

    print("Input:", input_tensor)
    print("CPU_Pooled_Output:", pooled_output)
    print("indices:", indices)
    print("TPU_Pooled_Output:", pooled_output_tpu.cpu())
    print("indices_tpu:", indices_tpu.cpu())

def max_pool2d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1):

    pooled_output, indices = torch.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation, return_indices=True)
    return pooled_output, indices

def test_avg_pool2d():
    input_tensor = torch.tensor(
    [[[[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]]]], dtype=torch.float32)
    input_tensor_tpu = input_tensor.clone().to(device)
    pooled_output = avg_pool2d(input_tensor, kernel_size=2)
    pooled_output_tpu = avg_pool2d(input_tensor_tpu, kernel_size=2)

    print("***************test_avg_pool2d begin*****************")
    print("Input:", input_tensor)
    print("CPU_Pooled Output:", pooled_output)
    print("TPU_Pooled_Output:", pooled_output_tpu.cpu())
    print("***************test_avg_pool2d end*****************")

def avg_pool2d(input, kernel_size, stride=1, padding=0):

    pooled_output = torch.nn.functional.avg_pool2d(input, kernel_size, stride, padding)
    return pooled_output

def test_adaptive_avg_pool2d():
    input_tensor = torch.tensor(
        [[[[1, 2, 3, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12],
           [13, 14, 15, 16]]]], dtype=torch.float32)
    input_tensor_tpu = input_tensor.clone().to(device)
    pooled_output = torch._adaptive_avg_pool2d(input_tensor, output_size=2)
    pooled_output_tpu = torch._adaptive_avg_pool2d(input_tensor_tpu, output_size=2)

    print("***************test_adaptive_avg_pool2d begin*****************")
    print("Input:", input_tensor)
    print("CPU_Pooled Output:", pooled_output)
    print("TPU_Pooled_Output:", pooled_output_tpu.cpu())
    print("***************test_adaptive_avg_pool2d end*****************")


if __name__ == "__main__":
    # test_max_pool2d_with_indices()
    test_avg_pool2d()
    test_adaptive_avg_pool2d()
