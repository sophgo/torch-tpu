import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    input_origin=torch.tensor([1, float('inf'), -2, float('-inf'), float('nan')])
    input_tpu=input_origin.to(device)
    
    
    output_cpu_prims=torch.ops.prims.isfinite(input_origin)
    output_tpu_prims=torch.ops.prims.isfinite(input_tpu).cpu()
    
    print("input_origin : ",input_origin)
    print("output_cpu_prims : ", output_cpu_prims)
    print("output_tpu_prims : ", output_tpu_prims)
    print("delta_prims : ",output_cpu_prims==output_tpu_prims)
    
if __name__ == "__main__":
    case1()