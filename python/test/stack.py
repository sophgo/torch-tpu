import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
device = "tpu"
def case1():
    Len = 200
    aa = torch.ones((5,5))
    StackL = []
    StackL_tpu = []
    for i in range(Len):
        a = torch.norm(aa, 2.0)
        StackL.append(a)
        StackL_tpu.append(a.to(device))
    o_cpu = torch.stack(StackL)
    o_tpu = torch.stack(StackL_tpu)
    print("cpu: ", o_cpu)
    print("tpu: ", o_tpu.cpu())

if __name__ == "__main__":
    case1()