import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
#from utils import compare_model_grad
import torch_tpu

device = 'tpu'

def case_Tensor():
    path = "case1_x.pt"
    B = 2
    H = 64
    W = 64
    x = torch.rand((B,H,W)).tpu()
    torch.save(x, path)
    
    l1 = torch.load(path, map_location="cpu")
    diff1 = l1 - x.cpu()
    print(torch.max(abs(diff1)))

    #TODO 
    # l2 = torch.load(path, map_location="tpu")
    # diff2 = l2 - x.cpu()
    # print(torch.max(abs(diff2)))

    # l3 = torch.load(path, map_location={"tpu":"cpu"})
    # diff3 = l3 - x.cpu()
    # print(torch.max(abs(diff3)))

    l4 = torch.load(path, map_location={"cpu":"tpu"})
    diff4 = l4 - x.cpu()
    print(torch.max(abs(diff4)))

    l5 = torch.load(path, map_location={"tpu:0":"tpu:1"})
    diff5 = l5 - x.cpu()
    print(torch.max(abs(diff5)))

def case_NNModule():
    IC = 4
    OC = 320
    K = 3
    net = nn.Conv2d(in_channels=IC, out_channels=OC, kernel_size=K, bias=True)
    net.to(device)
    torch.save(net, "case_net.bin")

if __name__ == "__main__":
    case_Tensor()
    #case_NNModule()