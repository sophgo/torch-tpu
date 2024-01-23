import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    input_origin=torch.rand(5,4,5,4)
    input_tpu=input_origin.to(device)

    output_cpu_aten=torch.ops.aten.replication_pad3d(input_origin,[2,3,1,1,2,3])
    output_tpu_aten=torch.ops.aten.replication_pad3d(input_tpu,[2,3,1,1,2,3]).cpu()

    print("input_origin : \n",input_origin)
    print("output_cpu_aten : \n", output_cpu_aten)
    print("output_tpu_aten : \n", output_tpu_aten)
    print("delta_aten : \n",torch.max((output_cpu_aten-output_tpu_aten)/output_cpu_aten))

if __name__ == "__main__":
    case1()