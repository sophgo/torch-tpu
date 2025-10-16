import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

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
    print("delta_aten : \n",torch.max((output_cpu_1-output_tpu_1.cpu())))
    print("****************test_gather end******************")

def test_index_select():
    input = torch.tensor(
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]], dtype=torch.float32)

    index = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    output_cpu = torch.index_select(input, -1, index)
    output_tpu = torch.index_select(input.to(device), -1, index.to(device))

    print("***************index_select begin*****************")
    print("input : ", input, input.size())
    print("index : ", index)
    print("output_cpu : ", output_cpu, "Shape:", output_cpu.shape)
    print("output_tpu : ", output_tpu.cpu(), "Shape:", output_tpu.shape)
    print("delta_aten : ", torch.max((output_cpu-output_tpu.cpu()).abs()))
    print("****************index_select end******************")



if __name__ == "__main__":
    test_gather()
    # test_index_select()