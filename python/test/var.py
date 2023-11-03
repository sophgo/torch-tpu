import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
device = "privateuseone:0"

def case1():
    left = -100.
    right = 100.
    shape = (23, 35, 22, 13)
    ori = left + (right - left) * torch.rand(*shape, dtype=torch.float)
    ori_tpu = ori.to(device)
    res_cpu = ori.var([0,1,2])
    res_tpu = ori_tpu.var([0,1,2]).cpu()

    print("ori : ", ori)
    print("cpu res : ", res_cpu)
    print("tpu res : ", res_tpu)
    print("ori shape :", ori.shape)
    print("cpu res shape :", res_cpu.shape)
    print("tpu res shape :", res_tpu.shape)
    print("max diff : ", torch.max(torch.div(torch.abs(res_cpu - res_tpu), res_cpu)))

if __name__ == "__main__" :
    case1()