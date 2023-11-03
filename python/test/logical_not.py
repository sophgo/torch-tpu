import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():

    a1 = torch.randint(0,3,(5, 5), dtype=torch.float32)
    a1_clone = a1.clone()
    a2 = a1.clone()
    a2_tpu = a2.to(device)


    a1.logical_not_()
    a2_tpu.logical_not_()
    print("origin: ",a1_clone)
    print("cpu : ", a1 )
    print("tpu : ", a2_tpu.cpu())



if __name__ == "__main__":
    case1()
