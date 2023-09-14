import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = "privateuseone:0"

def case1():
    # left = -100.
    # right = 100.
    # shape = (3, 2)
    # ori = left + (right - left) * torch.rand(*shape, dtype=torch.float)
    ori = torch.randint(1, 10, [3, 2, 3], dtype=torch.int)
    ori_tpu = ori.to(device)
    res_cpu = ori.repeat(2, 3, 5, 5)
    res_tpu = ori_tpu.repeat(2, 3, 5, 5).cpu()

    print("ori : ", ori)
    print("cpu res : ", res_cpu)
    print("tpu res : ", res_tpu)
    print("ori shape :", ori.shape)
    print("cpu res shape :", res_cpu.shape)
    print("tpu res shape :", res_tpu.shape)
    print("max diff : ", torch.max(torch.abs(res_cpu - res_tpu)))

if __name__ == "__main__" :
    case1()