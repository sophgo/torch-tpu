import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    input_origin=torch.tensor([4.5,5.4])
    
    input_tpu=input_origin.to(device)
    
    output_cpu_aten=torch.ops.aten.atan(input_origin)
    output_tpu_aten=torch.ops.aten.atan(input_tpu).cpu()
    output_cpu_prims=torch.ops.prims.atan(input_origin)
    output_tpu_prims=torch.ops.prims.atan(input_tpu).cpu()
    
    # print("input_origin : ",input_origin)
    print("output_cpu_aten : ", output_cpu_aten)
    print("output_tpu_aten : ", output_tpu_aten)
    print("delta_aten : ",(output_cpu_aten-output_tpu_aten)/output_cpu_aten)
    print("output_cpu_prims : ", output_cpu_prims)
    print("output_tpu_prims : ", output_tpu_prims)
    print("delta_prims : ",(output_cpu_prims-output_tpu_prims)/output_cpu_prims)
    
if __name__ == "__main__":
    case1()