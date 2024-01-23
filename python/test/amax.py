import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():

    input_origin=torch.randint(-10,10,(100,30,4,5,3)).to(torch.float32)
    input_tpu=input_origin.to(device)
    output_cpu=torch.amax(input_origin,dim=[0],keepdim=True)
    output_tpu=torch.amax(input_tpu,dim=[0],keepdim=True).cpu()

    print("input_origin : ",input_origin)
    print("output_cpu : ", output_cpu)
    print("output_tpu : ", output_tpu)
    print("delta : ",(output_cpu==output_tpu).all())


if __name__ == "__main__":
    case1()
    
