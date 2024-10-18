import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
device = "tpu"
def case1():
    a = torch.randn((5,5))
    a_tpu = a.to(device)
    
    o_cpu = torch.clamp(a, max=1, min=-1)
    o_tpu = torch.clamp(a_tpu, max=1, min=-1)
    print("cpu: ", o_cpu)
    print("tpu: ", o_tpu.cpu())

if __name__ == "__main__":
    case1()