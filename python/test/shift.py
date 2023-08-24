import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def test_shift_left():

    input = torch.tensor(
    [[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]],dtype=torch.uint8)
    shift = torch.tensor(
    [[0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2]],dtype=torch.uint8)
    shift_c =torch.tensor(2, dtype=torch.uint8)
    # shift_bcast = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)

    output_cpu = torch.ops.prims.shift_left(input,shift)
    output_tpu = torch.ops.prims.shift_left(input.to(device),shift.to(device))
    output_cpu_c = torch.ops.prims.shift_left(input,shift_c)
    output_tpu_c = torch.ops.prims.shift_left(input.to(device),shift_c)
    # output_tpu_bcast = torch.ops.prims.shift_left(input.to(device),shift_bcast.to(device))
    
    print("***************begint test_shift_left*****************")
    print("input : ", input)
    print("shift : ", shift)
    print("shift_c : ", shift_c)
    print("output_cpu : ", output_cpu )
    print("output_tpu : ", output_tpu.cpu())
    print("output_cpu_c : ", output_cpu_c )
    print("output_tpu_c : ", output_tpu_c.cpu())
    # print("output_tpu_bcast : ", output_tpu_bcast.cpu())

def test_shift_right_arithmetic():
    input = torch.tensor(
    [[-1, 2, 3, -4],
    [5, -6, -7, 8],
    [9, -10, -11, 12],
    [-13, 14, 15, -16]],dtype=torch.int32)
    shift = torch.tensor(
    [[0, 1, 2, -3],
    [1, 2, -3, 0],
    [2, -3, 0, 1],
    [-3, 0, 1, 2]],dtype=torch.int32)
    shift_c =torch.tensor(2, dtype=torch.int32)
    # shift_bcast = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)

    output_cpu = torch.ops.prims.shift_right_arithmetic(input,shift)
    output_tpu = torch.ops.prims.shift_right_arithmetic(input.to(device),shift.to(device))
    output_cpu_c = torch.ops.prims.shift_right_arithmetic(input,shift_c)
    output_tpu_c = torch.ops.prims.shift_right_arithmetic(input.to(device),shift_c)
    # output_tpu_bcast = torch.ops.prims.shift_right_arithmetic(input.to(device),shift_bcast.to(device))
    print("************begint test_shift_right_arithmetic**************")
    print("input : ", input)
    print("shift : ", shift)
    print("shift_c : ", shift_c)
    print("output_cpu : ", output_cpu )
    print("output_tpu : ", output_tpu.cpu())
    print("output_cpu_c : ", output_cpu_c )
    print("output_tpu_c : ", output_tpu_c.cpu())


if __name__ == "__main__":
    test_shift_left()
    test_shift_right_arithmetic()