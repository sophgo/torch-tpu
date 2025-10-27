import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy

from loguru import logger
import torch_tpu
torch.manual_seed(10001)
torch.set_printoptions(precision=20)
device = "tpu:0"

def cos_sim(vector_a, vector_b):
    vector_a = vector_a.reshape(-1)
    vector_b = vector_b.reshape(-1)
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    with np.errstate(invalid='ignore'):
        cos = np.nan_to_num(num / denom)
    sim = 0.5 + 0.5 * cos
    return sim

def cumsum_test():
    a = torch.randn((2,3,4,5))

    # 沿axis=0（行方向，垂直）累加
    print(a.shape)
    # out0 = torch.cumsum(a, dim=2)
    out0 = torch.ops.aten.cumsum(a, dim=2)
    # print(out0)

    b = a.reshape(6,4,5)
    print(b.shape)
    out1 = torch.cumsum(b, dim=1)
    # print(out1)
    out1 = out1.reshape((2,3,4,5))
    print(out0.all() == out1.all())

def cumsum1():
    import sys
    sys.float_repr = lambda x: format(x, '.15g')
    eps = 1e-3
    cos_thr = 0.99
    n = 1
    c = 1
    h = 40
    w = 1048
    x = torch.randn(h, w, dtype=torch.float32)
    x = torch.softmax(x, dim=-1) # condition 1 in tgi
    # x = torch.randint(0, 1024, (n, c, h, w), dtype=torch.int32) # condition 2 in tgi
    out0 = torch.ops.aten.cumsum(x, dim=-1, dtype=torch.float32)
    out_cpu = out0.flatten()
    print("cpu done")

    x_tpu = x.to(device)
    out_tpu = torch.cumsum(x_tpu, -1, dtype=torch.float32)
    out_tpu = out_tpu.cpu().flatten()
    print((out_cpu == out_tpu).all())
    cos_sim_val = cos_sim(out_cpu, out_tpu)
    out_diff = out_cpu - out_tpu
    ratio = abs(out_diff/torch.max(abs(out_cpu), abs(out_tpu) + 1e-6))
    key = torch.argmax(ratio)
    print("cos_sim_val: ", cos_sim_val)
    print("key : {}, max_diff : {}, tpu_out : {}, cpu_out : {}".format(key, ratio[key], out_tpu[key], out_cpu[key]))
    if cos_sim_val < cos_thr or ratio[key] > eps:
        print("cos_sim_val: ", cos_sim_val)
        print("max_diff : {}, tpu_out : {}, cpu_out : {}".format(ratio[key], out_tpu[key], out_cpu[key]))
        import pdb
        pdb.set_trace()
    else:
        print("compare success\n")


def cumsum2():
    for idx in range(3):
        print(f"The {idx}th stress test...")
        scores_cpu = torch.randn(1, 256).to(torch.bfloat16)
        probs_cpu = F.softmax(scores_cpu, dim=-1)
        probs_tpu = probs_cpu.to(device).to(torch.bfloat16)
        out= probs_tpu[0].cumsum(dim=-1).to("cpu")
        out_cpu = probs_cpu[0].cumsum(dim=-1)
        print(out[-1]==out_cpu[-1])

# 测试示例
if __name__ == "__main__":
    cumsum1()
    cumsum2()