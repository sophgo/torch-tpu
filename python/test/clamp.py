import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = "privateuseone"
def case1():
    aa = torch.randn((5,5))
    a = torch.norm(aa, 2.0)
    a_tpu = a.to(device)
    
    o_cpu = torch.clamp(a, max=0)
    o_tpu = torch.clamp(a_tpu, max=0)
    print("cpu: ", o_cpu)
    print("tpu: ", o_tpu.cpu())

if __name__ == "__main__":
    case1()