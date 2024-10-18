import torch
import random

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def test_squeeze():

    num_dims = random.randint(1, 4)  # random dimension number
    dim_sizes = [random.randint(1, 3) for _ in range(num_dims)] # random dimension size
    a1 = torch.rand(*dim_sizes)

    a2 = a1.clone()

    print("test_squeeze, origin ======")
    print(a1,"\n",a1.shape,"\n")
    res_cpu=torch.ops.aten.squeeze(a1)

    a2_tpu = a2.to(device)
    res_tpu = torch.ops.aten.squeeze(a2_tpu)

    print("cpu ======")
    print(res_cpu,"\n",res_cpu.shape,"\n")

    print("tpu ======")
    print(res_tpu.cpu(),"\n",res_tpu.cpu().shape,"\n")

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)


def test_squeeze_dim():

    a1 = torch.randn(3, 3, 1, 3)
    a2 = a1.clone()
    dim_random = random.randint(-a1.dim(), a1.dim() - 1) # get a random dim

    print("test_squeeze_dim, origin ======")
    print(a1,"\n",a1.shape,"\n")
    res_cpu=torch.ops.aten.squeeze.dim(a1, dim_random)

    a2_tpu = a2.to(device)
    res_tpu = torch.ops.aten.squeeze.dim(a2_tpu, dim_random)

    print("cpu ======")
    print(res_cpu,"\n",res_cpu.shape,"\n")

    print("tpu ======")
    print(res_tpu.cpu(),"\n",res_tpu.cpu().shape,"\n")

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)

def test_squeeze_dims():

    num_dims = random.randint(2, 4)  # random dimension number
    dim_sizes = [random.randint(1, 3) for _ in range(num_dims)] # random dimension size
    a1 = torch.rand(*dim_sizes)

    a2 = a1.clone()

    print("test_squeeze_dims, origin ======")
    print(a1,"\n",a1.shape,"\n")
    res_cpu=torch.ops.aten.squeeze.dims(a1, [0, 1])

    a2_tpu = a2.to(device)
    res_tpu = torch.ops.aten.squeeze.dims(a2_tpu, [0, 1])

    print("cpu ======")
    print(res_cpu,"\n",res_cpu.shape,"\n")

    print("tpu ======")
    print(res_tpu.cpu(),"\n",res_tpu.cpu().shape,"\n")

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)


if __name__ == "__main__":
    test_squeeze()
    test_squeeze_dim()
    test_squeeze_dims()