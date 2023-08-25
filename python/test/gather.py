import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def test_gather():

    input = torch.tensor(
    [[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]],dtype=torch.float32)
    index = torch.tensor(
    [[0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2]],dtype=torch.int64)
    # index = torch.tensor(
    # [[0, 1, 2, 3],
    # [0, 1, 2, 3],
    # [0, 1, 2, 3],
    # [0, 1, 2, 3]],dtype=torch.int64)
    
    # not support axis != -1 now 
    # output_cpu_0 = torch.gather(input, 0, index)
    # output_tpu_0 = torch.gather(input.to(device), 0, index.to(device))
    output_cpu_1 = torch.gather(input, 1, index)
    output_tpu_1 = torch.gather(input.to(device), 1, index.to(device))
    
    print("***************test_gather begin*****************")
    print("input : ", input)
    print("index : ", index)
    # print("output_cpu_0 : ", output_cpu_0 )
    # print("output_tpu_0 : ", output_tpu_0.cpu())
    print("output_cpu_1 : ", output_cpu_1 )
    print("output_tpu_1 : ", output_tpu_1.cpu())
    print("****************test_gather end******************")

if __name__ == "__main__":
    test_gather()