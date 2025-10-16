import torch
import torch.nn as nn
import torch.nn.functional as F
from test_utils import *

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    dim = 4
    shape = [10,10,200,10,2,4,5]
    dtype = torch.float32
    input_origin = (torch.rand(shape)*(-20000)+10000).to(dtype)
    input_tpu=input_origin.to(device)
    output_cpu,index_cpu=torch.ops.aten.min.dim(input_origin,dim=dim)
    output_tpu,index_tpu=torch.ops.aten.min.dim(input_tpu,dim=dim)
    output_tpu = output_tpu.cpu()
    index_tpu = index_tpu.cpu()
    # print("input_origin : ",input_origin)
    # print("output_cpu : ", output_cpu)
    # print("output_tpu : ", output_tpu)
    # print("output_cpu : ", index_cpu)
    # print("output_tpu : ", index_tpu)
    print("delta : ",(output_cpu==output_tpu).all(),(index_cpu==index_tpu).all())

def case2():
    dim = 1
    shape = [1,1024,1,2048]
    dtype = torch.float32
    input_origin = (torch.rand(shape)*(-20000)+10000).to(dtype)
    input_tpu=input_origin.to(device)
    keepdim = True
    out_shape = list(input_origin.size())
    out_shape[dim] = 1
    values_out_cpu = torch.empty(out_shape, dtype=input_origin.dtype)
    indices_out_cpu = torch.empty(out_shape, dtype=torch.int64)
    values_out_tpu = torch.empty(out_shape, dtype=input_origin.dtype, device=device)
    indices_out_tpu = torch.empty(out_shape, dtype=torch.int32, device=device)

    output_cpu, index_cpu = torch.ops.aten.min.dim_min(
        input_origin, dim=dim, keepdim=keepdim, min=values_out_cpu, min_indices=indices_out_cpu)
    output_tpu, index_tpu = torch.ops.aten.min.dim_min(
        input_tpu, dim=dim, keepdim=keepdim, min=values_out_tpu, min_indices=indices_out_tpu)
    output_tpu = output_tpu.cpu()
    index_tpu = index_tpu.cpu()
    # print("input_origin : ",input_origin)
    # print("output_cpu : ", output_cpu)
    # print("output_tpu : ", output_tpu)
    # print("output_cpu : ", index_cpu)
    # print("output_tpu : ", index_tpu)
    print("delta : ",(output_cpu==output_tpu).all(),(index_cpu==index_tpu).all())

if __name__ == "__main__":
    case1()