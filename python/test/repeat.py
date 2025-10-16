import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
device = "tpu:0"

def case1():
    dtypes = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int16,
        torch.int8
    ]

    for dtype in dtypes:
        ori = torch.randint(1, 10, [256, 1, 1, 256], dtype=dtype)
        ori_tpu = ori.to(device)
        res_cpu = ori.repeat(1, 1, 1, 2)
        res_tpu = ori_tpu.repeat(1, 1, 1, 2).cpu()

        # print("ori : ", ori)
        # print("cpu res : ", res_cpu)
        # print("tpu res : ", res_tpu)
        # print("ori shape :", ori.shape)
        # print("cpu res shape :", res_cpu.shape)
        # print("tpu res shape :", res_tpu.shape)
        print("max diff : ", torch.max(torch.abs(res_cpu - res_tpu)))

if __name__ == "__main__" :
    case1()