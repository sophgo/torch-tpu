import torch
import torch.nn as nn
import torch.nn.functional as F


torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    input_origin=torch.rand(2,3,4,5,3)
    input_tpu=input_origin.to(device)
    output_cpu=torch.argmax(input_origin,dim=2)
    output_tpu=torch.argmax(input_tpu,dim=2).cpu()
    print("input_origin : ",input_origin)
    print("output_cpu : ", output_cpu.shape)
    print("output_tpu : ", output_tpu.shape)
    print("delta : ",output_cpu==output_tpu)

if __name__ == "__main__":
    case1()


