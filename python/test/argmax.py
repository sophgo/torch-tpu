import torch
import torch.nn as nn
import torch.nn.functional as F


torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    dim = None
    input_origin=torch.rand(1,1,3840,1)
    input_tpu=input_origin.to(device)
    output_cpu=torch.argmax(input_origin,dim=dim)
    output_tpu=torch.argmax(input_tpu,dim=dim).cpu()
    # print("input_origin : ",input_origin)
    print("output_cpu : ", output_cpu.shape)
    print("output_tpu : ", output_tpu.shape)
    print("delta : ",(output_cpu==output_tpu).all())

if __name__ == "__main__":
    case1()


