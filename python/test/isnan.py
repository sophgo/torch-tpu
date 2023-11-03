import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    input_origin=torch.tensor([1, float('inf'), -2, float('-inf'), float('nan')])
    input_tpu=input_origin.to(device)
    
    
    output_cpu_aten=torch.ops.aten.isnan(input_origin)
    output_tpu_aten=torch.ops.aten.isnan(input_tpu).cpu()
    
    print("input_origin : ",input_origin)
    print("output_cpu_aten : ", output_cpu_aten)
    print("output_tpu_aten : ", output_tpu_aten)
    print("delta_aten : ",output_cpu_aten==output_tpu_aten)
    
if __name__ == "__main__":
    case1()