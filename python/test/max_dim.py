import torch
import torch.nn as nn
import torch.nn.functional as F


torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    input_origin=torch.rand(3,4,5)
    input_tpu=input_origin.to(device)
    output_cpu,index_cpu=torch.ops.aten.max.dim(input_origin,True)
    output_tpu,index_tpu=torch.ops.aten.max.dim(input_tpu,True)
    output_tpu = output_tpu.cpu()
    index_tpu = index_tpu.cpu()
    # print("input_origin : ",input_origin)
    # print("output_cpu : ", output_cpu)
    # print("output_cpu : ", index_cpu)
    # print("output_tpu : ", output_tpu)
    # print("output_tpu : ", index_tpu)
    print("delta : ",(output_cpu==output_tpu).all(),(index_cpu==index_tpu).all())

if __name__ == "__main__":
    case1()


