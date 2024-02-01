import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():

    input_origin=torch.rand(4,5,6,3)
    input_tpu=input_origin.to(device)
    output_cpu=torch.amin(input_origin,dim=[1,2,3],keepdim=True)
    output_tpu=torch.amin(input_tpu,dim=[1,2,3],keepdim=True).cpu()

    print("input_origin : ",input_origin)
    print("output_cpu : ", output_cpu)
    print("output_tpu : ", output_tpu)
    print("delta : ",(output_cpu-output_tpu)/output_cpu)


if __name__ == "__main__":
    case1()
    
