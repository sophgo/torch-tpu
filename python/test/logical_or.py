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
    a1_tpu = a1.to(device)
    a1_clone = a1.clone()


    a1_clone.logical_or_(a2)
    a1_tpu.logical_or_(a2.to(device))
    print("a1: ", a1)
    print("a2 : ", a2)
    print("cpu : ", a1_clone )
    print("tpu : ", a1_tpu.cpu())



if __name__ == "__main__":
    case1()
