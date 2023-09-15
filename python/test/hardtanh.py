import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = "privateuseone:0"

def case1():
    left = -1000.
    right = 1000.
    shape = (33, 52, 111)
    ori = left + (right - left) * torch.rand(*shape, dtype=torch.float)
    # ori = torch.randint(-1000, 1000, [35, 151], dtype=torch.int)
    ori_tpu = ori.to(device)
    res_cpu = F.hardtanh(ori, -535, 353)
    res_tpu = F.hardtanh(ori_tpu, -535, 353).cpu()

    print("ori : ", ori)
    print("cpu res : ", res_cpu)
    print("tpu res : ", res_tpu)
    print("max diff : ", torch.max(torch.abs(res_cpu - res_tpu)))

if __name__ == "__main__" :
    case1()