import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
#from utils import compare_model_grad
import torch_tpu

def is_torch_device():
    device = torch.device("privateuseone", 0)
    tensor = torch.randn((10)).to(device)
    net = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, bias=True).to(device)

    d2 = next(net.parameters()).device
    d3 = tensor.device
    print( isinstance(device, torch.device))
    print( isinstance(d2, torch.device))
    print( isinstance(d3, torch.device))

if __name__ == "__main__":
    is_torch_device()