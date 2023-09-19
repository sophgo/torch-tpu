import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
device = "privateuseone:0"
torch.set_printoptions(precision=30)
def case1():
    input0_origin=torch.rand(4,1,4,5)
    input1_origin=torch.rand(4,1,5)
    input2_origin=torch.rand(1,4,1)
    input3_origin=torch.tensor([[1.0,-1.0,0.0,-0.0,float('inf'),-float('inf'),float('nan')]]*2)
    input4_origin=torch.tensor([10.0]*7)

    output_cpu=torch.nextafter(input0_origin,input1_origin)
    output_tpu=torch.nextafter(input0_origin.to(device),input1_origin.to(device)).cpu()
    
        
    print('cpu :',output_cpu)
    print('tpu :',output_tpu)
    print('delta :',(output_cpu==output_tpu))

if __name__ == "__main__":
    case1()