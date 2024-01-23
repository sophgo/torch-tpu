import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
device = "tpu:0"
torch.set_printoptions(precision=8)

def case1():
    left = -100.
    right = 100.
    shape = (3,2,4)
    ori = left + (right - left) * torch.rand(*shape).float()
    # ori = torch.tensor(10)
    ori_tpu = ori.to(device)
    res_cpu = torch.constant_pad_nd(ori, [1,0,0,2,0,3], 1)
    res_tpu = torch.constant_pad_nd(ori_tpu, [1,0,0,2,0,3], 1).cpu()

    print("ori : ", ori)
    print("cpu res : ", res_cpu)
    print("tpu res : ", res_tpu)
    print("ori shape :", ori.shape)
    print("cpu res shape :", res_cpu.shape)
    print("tpu res shape :", res_tpu.shape)
    print("max diff : ", torch.max(torch.abs(res_cpu - res_tpu)))

if __name__ == "__main__" :
    case1()