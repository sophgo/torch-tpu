import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
device = "tpu"

def case_topk():
    left = -1000.
    right = 1000.
    shape = (33, 55, 11, 155)
    # shape = (2, 3, 4)
    ori = left + (right - left) * torch.rand(*shape, dtype=torch.float)
    # ori = torch.randint(1, 10, (3, 11), dtype=torch.int)
    # ori = torch.tensor([2, 2, 3, 3, 4, 5, 6, 6, 6], dtype=torch.float)
    print(ori.shape, ori.dim())
    res_cpu_value, res_cpu_index = torch.topk(ori, k=5)
    res_tpu_value, res_tpu_index = torch.topk(ori.to(device), k=5)

    # print("ori : ", ori)
    # print("cpu res value : ", res_cpu_value)
    # print("tpu res value : ", res_tpu_value.cpu())
    # print("cpu res index : ", res_cpu_index)
    # print("tpu res index : ", res_tpu_index.cpu())
    print("max value diff : ", torch.max(torch.abs(res_cpu_value - res_tpu_value.cpu())))
    print("max index diff : ", torch.max(torch.abs(res_cpu_index - res_tpu_index.cpu())))

def case_sort():
    left = -1000.
    right = 1000.
    shape = (33, 55, 11, 115)
    ori = left + (right - left) * torch.rand(*shape, dtype=torch.float)

    res_cpu_value, res_cpu_index = torch.sort(ori, stable=True)
    res_tpu_value, res_tpu_index = torch.sort(ori.to(device))

    # print("ori : ", ori)
    # print("cpu res value : ", res_cpu_value)
    # print("tpu res value : ", res_tpu_value.cpu())
    print("cpu res index : ", res_cpu_index)
    print("tpu res index : ", res_tpu_index.cpu())
    print("max value diff : ", torch.max(torch.abs(res_cpu_value - res_tpu_value.cpu())))
    print("max index diff : ", torch.max(torch.abs(res_cpu_index - res_tpu_index.cpu())))

if __name__ == "__main__" :
    # case_topk()
    case_sort()