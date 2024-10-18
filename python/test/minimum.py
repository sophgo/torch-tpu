import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"

def case1():
    input0_origin=torch.randint(1,100,(5,3,1,5),dtype= torch.int)
    input1_origin=torch.randint(1,100,(5,5),dtype= torch.int)
    input2_origin=torch.rand(1,4,1)
    input3_origin=torch.tensor([[2,1,0]]*20).to(torch.float32)
    input4_origin=torch.tensor(1).to(torch.float32)
    output_cpu=torch.ops.aten.minimum(input0_origin,input1_origin)
    output_tpu=torch.ops.aten.minimum(input0_origin.to(device),input1_origin.to(device)).cpu()
    # output_tpu=torch.minimum(input3_origin.to(device),input4_origin.to(device)).cpu()
    # output_tpu_r=torch.minimum(input4_origin.to(device),input3_origin.to(device)).cpu()
    print('cpu :',output_cpu)
    print('tpu :',output_tpu)
    # print('tpu_r :',output_tpu_r.dtype)
    print('delta :',(output_cpu==output_tpu).all())


if __name__ == "__main__":
    case1()