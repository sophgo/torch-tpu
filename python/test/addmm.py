import torch

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

if __name__ == "__main__":
    device = "privateuseone"

    m = 2
    k = 3
    n = 3

    mat = torch.rand(m, n)
    mat1 = torch.rand(m, k)
    mat2 = torch.rand(k, n)

    mat_tpu = mat.to(device)
    mat1_tpu = mat1.to(device)
    mat2_tpu = mat2.to(device)

    res_cpu = torch.addmm(mat, mat1, mat2)
    res_tpu = torch.addmm(mat_tpu, mat1_tpu, mat2_tpu)

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
