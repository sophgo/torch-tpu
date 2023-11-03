import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():
    dtype = torch.int32
    input0_origin=torch.randint(-100,100,(19,1,4,5)).to(dtype)
    input1_origin=torch.randint(-100,100,(1,4,5)).to(dtype)
    input2_origin=torch.rand(1,4,1)
    input3_origin=torch.tensor(-0.0)
    input4_origin=torch.tensor([1,0,-0.0,-1,float('inf'),-float('inf'),float('nan'),-float('nan')])

    output_cpu=torch.fmin(input0_origin,input1_origin)
    output_tpu=torch.fmin(input0_origin.to(device),input1_origin.to(device)).cpu()
    output_tpu_r=torch.fmin(input0_origin.to(device),input1_origin.to(device)).cpu()
    print('cpu :',output_cpu)
    print('tpu :',output_tpu)
    # print('tpu_r :',output_tpu_r)
    print('delta :',(output_cpu==output_tpu).all())
    print('delta :',(output_cpu==output_tpu_r).all())


if __name__ == "__main__":
    case1()