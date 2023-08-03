import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    length = 2
    end = 65536*64*10
    out1=torch.arange(0, end, length,device="cpu")
    print("cpu: ", out1)

    # TPU: just support arange(int,int,int) currentlly.
    out2=torch.arange(0, end, length,device=device)
    print("Tpu: ", out2.cpu())

    diff = torch.sum( out1 - out2.cpu() )
    print( "Difference: ", diff)




if __name__ == "__main__":
    case1()