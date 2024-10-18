import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    dtype = torch.float32
    input_origin=torch.zeros((2,4,5,6)).to(dtype)
    src_origin = torch.ones(2,4,2,6).to(dtype)
    input_tpu=input_origin.to(device)
    src_tpu=src_origin.to(device)

    output_cpu_aten=torch.ops.aten.slice_scatter(input_origin,src_origin,2,0,3,step=2)
    output_tpu_aten=torch.ops.aten.slice_scatter(input_tpu,src_tpu,2,0,3,step=2).cpu()


    print("input_origin : \n",input_origin)
    print("output_cpu_aten : \n", output_cpu_aten)
    print("output_tpu_aten : \n", output_tpu_aten)
    print("delta_aten : \n",torch.max((output_cpu_aten-output_tpu_aten)))
if __name__ == "__main__":
    case1()