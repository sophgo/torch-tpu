import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case_baddbmmcast_1():
    M = torch.randn(3, 5)
    batch1 = torch.randn(10, 3, 4)
    batch2 = torch.randn(10, 4, 5)
    M_tpu = M.to(device).half()
    batch1_tpu = batch1.to(device).half()
    batch2_tpu = batch2.to(device).half()
    alpha = 2.0
    beta = 3.0
    o = torch.baddbmm(M, batch1, batch2, beta=beta, alpha=alpha)# baddbmm not support half
    o_t = torch.baddbmm(M_tpu, batch1_tpu, batch2_tpu, beta=beta, alpha=alpha).float()
    diff = o - o_t.cpu()
    print(torch.max(torch.abs(diff)))

def case_baddbmmcast():
    M = torch.randn(10, 3, 5)
    batch1 = torch.randn(10, 3, 4)
    batch2 = torch.randn(10, 4, 5)
    M_tpu = M.to(device)
    batch1_tpu = batch1.to(device)
    batch2_tpu = batch2.to(device)
    alpha = 1.0
    beta = 1.0
    o = torch.baddbmm(M, batch1, batch2, beta=beta, alpha=alpha)
    o_t = torch.baddbmm(M_tpu, batch1_tpu, batch2_tpu, beta=beta, alpha=alpha)
    diff = o - o_t.cpu()
    print(torch.max(torch.abs(diff)))

def case_baddbmmcast_0():
    M = torch.randn(10, 3, 5)
    batch1 = torch.randn(10, 3, 4)
    batch2 = torch.randn(10, 5, 4)
    M_tpu = M.to(device)
    batch1_tpu = batch1.to(device)
    batch2_tpu = batch2.to(device)
    alpha = 1.0
    beta = 1.0
    o = torch.baddbmm(M, batch1, batch2.transpose(1,2), beta=beta, alpha=alpha)
    o_t = torch.baddbmm(M_tpu, batch1_tpu, batch2_tpu.transpose(1,2), beta=beta, alpha=alpha)
    diff = o - o_t.cpu()
    print(torch.max(torch.abs(diff)))

if __name__ == "__main__":
    #case1()
    case_baddbmmcast()
    case_baddbmmcast_0()
    case_baddbmmcast_1()