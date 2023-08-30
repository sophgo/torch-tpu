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
    input4_origin=torch.tensor([1,0,-0.0,-1,float('inf'),-float('inf'),float('nan'),-float('nan')])

    output_cpu=torch.fmin(input3_origin,input4_origin)
    output_tpu=torch.fmin(input3_origin.to(device),input4_origin.to(device)).cpu()
    output_tpu_r=torch.fmin(input4_origin.to(device),input3_origin.to(device)).cpu()
    print('cpu :',output_cpu)
    print('tpu :',output_tpu)
    print('tpu_r :',output_tpu_r)
    print('delta :',(output_cpu-output_tpu)/output_cpu)


if __name__ == "__main__":
    case1()