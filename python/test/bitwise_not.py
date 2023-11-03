import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    input_origin=torch.randint(-100,100,(10,50,20,10,10,10,10)).to(torch.int32)
    input_tpu=input_origin.to(device)

    output_cpu_aten=torch.ops.aten.bitwise_not(input_origin)
    output_tpu_aten=torch.ops.aten.bitwise_not(input_tpu).cpu()
    output_cpu_prims=torch.ops.prims.bitwise_not(input_origin)
    output_tpu_prims=torch.ops.prims.bitwise_not(input_tpu).cpu()
    
    # print("input_origin : ",input_origin)
    # print("output_cpu_aten : ", output_cpu_aten)
    # print("output_tpu_aten : ", output_tpu_aten)
    print("delta_aten : ",(output_cpu_aten==output_tpu_aten).all())
    # print("output_cpu_prims : ", output_cpu_prims)
    # print("output_tpu_prims : ", output_tpu_prims)
    print("delta_prims : ",(output_cpu_prims==output_tpu_prims).all())
    
if __name__ == "__main__":
    case1()