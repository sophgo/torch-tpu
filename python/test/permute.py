import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=8)
device = "tpu:0"

def case1(use_fp16 = False):

    # ori = torch.randint(-10, 10, (18, 5, 8, 118), dtype=torch.int32)
    ori = -10 + (10 - (-10)) * torch.rand(133, 8, 256)
    ori_tpu = ori.to(device)
    # cpu_res = torch.permute(ori, [1, 2, 0, 3])
    # tpu_res = torch.permute(ori_tpu, [1, 2, 0, 3]).cpu()
    cpu_res = ori.permute( 1, 2, 0)
    tpu_res = ori_tpu.permute( 1, 2, 0).cpu()

    print("ori : ", ori)
    print("cpu res : ", cpu_res)
    print("tpu res : ", tpu_res)
    print("max diff : ", torch.max(torch.abs(cpu_res - tpu_res)))

if __name__ == "__main__":
    case1()