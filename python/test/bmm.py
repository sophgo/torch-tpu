import torch

import torch_tpu
torch.manual_seed(1000)

if __name__ == "__main__":
    device = "tpu"

    b = 2
    m = 2
    k = 3
    n = 3

    mat1 = torch.rand(b, m, k)
    mat2 = torch.rand(b, k, n)

    mat1_tpu = mat1.to(device)
    mat2_tpu = mat2.to(device)

    res_cpu = torch.bmm(mat1, mat2)
    res_tpu = torch.bmm(mat1_tpu, mat2_tpu)

    print("cpu ======")
    print(res_cpu)
    print("tpu ======")
    print(res_tpu.cpu())

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)
    print("cpu:", res_cpu.flatten()[idx])
    print("tpu:", res_tpu.cpu().flatten()[idx])
