import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tpu

torch.set_printoptions(precision=6)
device = "tpu"

def case1():
    step = 1
    end = 1
    out1=torch.arange(0, end, step, device="cpu")
    print("cpu: ", out1)

    # TPU: just support arange(int,int,int) currentlly.
    out2=torch.arange(0, end, step, device=device)
    print("Tpu: ", out2.cpu())

    diff = torch.sum( out1 - out2.cpu() )
    print( "Difference: ", diff)

def case_dtype():
    step = 3
    end = 8
    out1=torch.arange(0, end, step, dtype = torch.float32, device="cpu")
    print("cpu: ", out1)

    # TPU: just support arange(int,int,int) currentlly.
    out2=torch.arange(0, end, step, dtype = torch.float32, device=device)
    print("Tpu: ", out2.cpu())

    diff = torch.sum( out1 - out2.cpu() )
    print( "Difference: ", diff)
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    #case1()
    case_dtype()