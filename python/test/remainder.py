import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    x = torch.randn(35).view(5,7)[:,2:4]
    x_tpu = x.to(device)
    y = x % 1
    y_tpu = x_tpu % 1
    import pdb; pdb.set_trace()

def case2():
    x = torch.randn(35).view(5,7)[:,2:4] * 13
    x = x.int()
    x_tpu = x.to(device)
    y = x % 2
    y_tpu = x_tpu % 2
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    #case1()
    case2()