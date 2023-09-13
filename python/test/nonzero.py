import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def test_nonzero():
    shape = []
    for i in range(1, 9):
        shape.append(i%5 + 3)
        input = torch.randint(10, shape, dtype=torch.int)
        input[input < 5] = 0 
        
        output_cpu = torch.nonzero(input)
        output_tpu = torch.nonzero(input.to(device)).cpu()

        print("input shape: ", shape)
        # print("cpu out: ", output_cpu)
        # print("tou out: ", output_tpu)
        print("diff abs sum: ", sum(abs(output_cpu - output_tpu)))


if __name__ == '__main__':
    test_nonzero()