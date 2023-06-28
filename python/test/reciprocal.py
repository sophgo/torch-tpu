import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = "privateuseone"
def case1():
    aa = torch.ones((5,5))
    a = torch.norm(aa, 2.0)
    a_tpu = a.to(device)
    b = 1.0

    o_cpu = b / (a + 1e-6)
    o_tpu = b / (a_tpu + 1e-6)

    print("cpu: ", o_cpu)
    print("tpu: ", o_tpu.cpu())

if __name__ == "__main__":
    case1()