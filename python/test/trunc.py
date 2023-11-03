import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
device = "privateuseone"
def case1():
    left = -1000.
    right = 1000.
    shape = (33, 55, 355)
    ori = left + (right - left) * torch.rand(*shape)
    ori_tpu = ori.to(device)
    res_cpu = torch.trunc(ori)
    res_tpu = torch.trunc(ori_tpu).cpu()

    print("ori : ", ori)
    print("cpu res : ", res_cpu)
    print("tpu res : ", res_tpu)
    print("max diff : ", torch.max(torch.abs(res_cpu - res_tpu)))

if __name__ == "__main__" :
    case1()