import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = "privateuseone"


def case1():
    # ori = torch.rand(125, 235, 355)
    # ori = torch.randint(1, 10, (125, 235, 355), dtype = torch.int8)
    ori = torch.tensor([0.0, 4, float("inf"), float("nan")])
    ori_tpu = ori.to(device)
    res_cpu = torch.reciprocal(ori)
    res_tpu = torch.reciprocal(ori_tpu).cpu()

    print("ori : ", ori)
    print("cpu res : ", res_cpu)
    print("tpu res : ", res_tpu)
    print("diff : ", torch.max(torch.div(torch.abs(res_cpu - res_tpu), res_cpu)))


if __name__ == "__main__":
    case1()
