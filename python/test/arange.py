import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    
    out=torch.arange(1,3,0.1,device=device)
    print("Tpu: ", out.cpu())
    out2=torch.arange(1,2,1,device="cpu")
    print("cpu: ", out2)



if __name__ == "__main__":
    case1()