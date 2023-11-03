import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
device = "privateuseone"

def case1():
    left = -1000.
    right = 1000.
    shape = (33, 55, 11, 155)
    ori = left + (right - left) * torch.rand(*shape, dtype=torch.float)
    # ori = torch.randint(1, 10, (3, 11), dtype=torch.int)
    ori_tpu = ori.to(device)
    res_cpu_value, res_cpu_index = torch.topk(ori, k=5)
    res_tpu_value, res_tpu_index = torch.topk(ori_tpu, k=5)

    print("ori : ", ori)
    print("cpu res value : ", res_cpu_value)
    print("tpu res value : ", res_tpu_value.cpu())
    print("cpu res index : ", res_cpu_index)
    print("tpu res index : ", res_tpu_index.cpu())
    print("max value diff : ", torch.max(torch.abs(res_cpu_value - res_tpu_value.cpu())))
    # print("max index diff : ", torch.max(torch.abs(res_cpu_index - res_tpu_index.cpu())))

if __name__ == "__main__" :
    case1()