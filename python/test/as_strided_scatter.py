import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def test():
    self = torch.zeros(2, 16, dtype=torch.float32)
    src = torch.arange(1, 21, dtype=torch.float32).view(4, 5)

    out_cpu = torch.ops.prims.as_strided_scatter(self, src, (4, 5), (1, 2), 0)
    print("out_cpu")
    print(out_cpu)
    # print(self)

    out_tpu = torch.as_strided_scatter(self.to(device), src.to(device), (4, 5), (1, 2), 0)
    print("out_tpu")
    print(out_tpu.cpu())
    print("特别说明:该算子在cpu端和cuda端，当赋值位置发生重叠时行为不一致。tpu实现与cuda保持一致。")

if __name__ == "__main__":
    test()