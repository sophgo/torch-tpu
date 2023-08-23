import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():
    input0_origin=torch.rand(4,1,4,5)
    input1_origin=torch.rand(4,1,4,5)
    input2_origin=torch.rand(1,4,1)
    input3_origin=torch.tensor(1.0)
    input4_origin=torch.tensor(2.0)

    output_cpu=torch.minimum(input0_origin,input2_origin)
    output_tpu=torch.minimum(input0_origin.to(device),input2_origin.to(device)).cpu()
    print('cpu :',output_cpu)
    print('tpu :',output_tpu)
    print('delta :',(output_cpu-output_tpu)/output_cpu)


if __name__ == "__main__":
    case1()