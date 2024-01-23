import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"
torch.set_printoptions(precision=10)
def case1(dtype):
    
    input0_origin=(torch.rand(4,1,4,5)*10-5).to(dtype)
    input1_origin=(torch.rand(4,1,4,5)*10-5).to(dtype)
    input2_origin=torch.rand(1,4,1).to(dtype)
    input4_origin=torch.tensor([[-1,1] * 8] *2).to(dtype)
    input3_origin=torch.tensor([0,-0,1,-1,0,-0,1,-1,0,-0,1,-1,0,-0,1,-1]).to(dtype)

    output_cpu=torch.nextafter(input3_origin,input4_origin)
    print('cpu :',output_cpu)
    # import struct
    # print('cpu :',[hex(struct.unpack('!I', struct.pack('!f', float(value)))[0]) for value in output_cpu])
    # output_cpu_r=torch.nextafter(input4_origin,input3_origin)
    # print('cpu :',output_cpu_r.dtype)

    output_tpu=torch.nextafter(input3_origin.to(device),input4_origin.to(device)).cpu()
    
    
    
    
    print('tpu :',output_tpu)
    print('delta :',(output_cpu==output_tpu).all())
 

if __name__ == "__main__":
    # print("************FP32***********")
    # dtype = torch.float32
    # case1(dtype)
    # print("************BF16***********")
    dtype = torch.bfloat16
    case1(dtype)