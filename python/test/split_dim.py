import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def test_split_dim():
    """
    args: [input, dim, length]
    e.g. input.shape = (s0, s1, s2)
    dim = 0 -> ishape = (length, s0/length, s1, s2)
    dim = 1 -> ishape = (s0, length, s1/length, s2)
    dim = 2 -> ishape = (s0, s1, length, s2/length)
    """
    input = torch.randn(4, 4, 4)
    
    length = torch.tensor([2])
    output_cpu_0 = torch.ops.prims.split_dim(input, 0, 2)
    output_tpu_0 = torch.ops.prims.split_dim(input.to(device), 0, length.to(device))
    output_cpu_1 = torch.ops.prims.split_dim(input, 1, 2)
    output_tpu_1 = torch.ops.prims.split_dim(input.to(device), 1, 2)
    
    print("***************test_split_dim begin*****************")
    print("input : ", input)
    print("output_cpu_0 : ", output_cpu_0.shape, output_cpu_0)
    print("output_tpu_0 : ", output_tpu_0.cpu().shape, output_tpu_0.cpu())
    print("output_cpu_1 : ", output_cpu_1.shape, output_cpu_1)
    print("output_tpu_1 : ", output_tpu_1.cpu().shape, output_tpu_1.cpu())
    print("****************test_split_dim end******************")

if __name__ == "__main__":
    test_split_dim()