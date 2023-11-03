import torch
import random

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def test_unsqueeze():

    a1 = torch.randn(1, 3, 1, 3)
    a2 = a1.clone()
    dim_random = random.randint(-a1.dim() - 1, a1.dim()) # get a random dim

    print("test_unsqueeze, origin ======")
    print(a1,"\n",a1.shape,"\n")
    res_cpu=torch.ops.aten.unsqueeze(a1, dim_random)

    a2_tpu = a2.to(device)
    res_tpu = torch.ops.aten.unsqueeze(a2_tpu, dim_random)

    print("cpu ======")
    print(res_cpu,"\n",res_cpu.shape,"\n")

    print("tpu ======")
    print(res_tpu.cpu(),"\n",res_tpu.cpu().shape,"\n")

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)

if __name__ == "__main__":
    test_unsqueeze()
