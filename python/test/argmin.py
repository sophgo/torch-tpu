import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    dim = None
    input_origin=torch.rand(2,4,2,4,4,6)
    input_tpu=input_origin.to(device)
    output_cpu=torch.argmin(input_origin,dim=dim)
    output_tpu=torch.argmin(input_tpu,dim=dim).cpu()
    # print("input_origin : ",input_origin)
    print("output_cpu : ", output_cpu.shape)
    print("output_tpu : ", output_tpu.shape)
    print("delta : ",(output_cpu==output_tpu).all())

if __name__ == "__main__":
    case1()

