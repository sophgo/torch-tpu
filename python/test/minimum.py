import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
device = "privateuseone:0"

def case1():
    input0_origin=torch.randint(-100,100,(200,1,4,1)).to(torch.float32)
    input1_origin=torch.randint(-100,100,(1,5)).to(torch.bfloat16)
    input2_origin=torch.rand(1,4,1)
    input3_origin=torch.tensor([[2,1,0]]*20).to(torch.float32)
    input4_origin=torch.tensor(1).to(torch.float32)
    output_cpu=torch.ops.aten.minimum(input3_origin,input3_origin)
    output_tpu=torch.ops.aten.minimum(input3_origin.to(device),input4_origin.to(device)).cpu()
    # output_tpu=torch.minimum(input3_origin.to(device),input4_origin.to(device)).cpu()
    # output_tpu_r=torch.minimum(input4_origin.to(device),input3_origin.to(device)).cpu()
    print('cpu :',output_cpu)
    print('tpu :',output_tpu)
    # print('tpu_r :',output_tpu_r.dtype)
    print('delta :',(output_cpu==output_tpu).all())


if __name__ == "__main__":
    case1()