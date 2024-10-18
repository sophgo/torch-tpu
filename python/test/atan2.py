import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"

def case1():
    input0_origin=torch.randint(-100,100,(19,1,4,5)).to(torch.float32)
    input1_origin=torch.randint(-100,100,(19,1,4,5)).to(torch.float32)
    input2_origin=torch.rand(1,4,1)
    input3_origin=torch.tensor(3.0).to(torch.bfloat16)
    input4_origin=torch.tensor([1.0,0.0,-0.0,-1.0]*100).to(torch.bfloat16)

    output_cpu=torch.atan2(input0_origin,input1_origin)
    output_tpu=torch.atan2(input0_origin.to(device),input1_origin.to(device)).cpu()
    print('cpu :',output_cpu)
    print('tpu :',output_tpu)
    print('delta :',torch.min((output_cpu-output_tpu)/output_cpu))

if __name__ == "__main__":
    case1()