import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():

    a1 = torch.randint(0,3,(5, 5), dtype=torch.float32)
    a2 = torch.randint(0,3,(5, 5), dtype=torch.float32)
    a2_tpu = a2.to(device)
    a3 = True
    a4 = torch.zeros((5, 5), dtype=torch.float32)


    a2.logical_and_(a4)
    a2_tpu.logical_and_(a4.to(device))
    print("origin: ",a1)
    print("cpu : ", a2 )
    print("tpu : ", a2_tpu.cpu())



if __name__ == "__main__":
    case1()