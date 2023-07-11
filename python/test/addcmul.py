import torch

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

if __name__ == "__main__":
    device = "privateuseone"

    m = 2
    n = 2

    t = torch.rand(m, n)
    t1 = torch.rand(m, n)
    t2 = torch.rand(m, n)

    t_tpu = t.to(device)
    t1_tpu = t1.to(device)
    t2_tpu = t2.to(device)

    res_cpu = torch.addcmul(t, t1, t2)
    res_tpu = torch.addcmul(t_tpu, t1_tpu, t2_tpu)

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
