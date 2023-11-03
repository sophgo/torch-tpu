import torch

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)

def case_broadcast():
    device = "privateuseone"
    N = 1
    C = 512
    H = 64
    W = 64

    inp = torch.randn(N, C, 1, 1)
    t1 = torch.randn(C, H, W)
    t2 = torch.randn(N, C, 1, 1)

    inp_tpu = inp.to(device)
    t1_tpu = t1.to(device)
    t2_tpu = t2.to(device)

    #test FP16
    # inp_tpu = inp.half().to(device)
    # t1_tpu = t1.half().to(device)
    # t2_tpu = t2.half().to(device)

    res_cpu = torch.addcmul(inp, t1, t2)
    res_tpu = torch.addcmul(inp_tpu, t1_tpu, t2_tpu)
    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)
    print("cpu:", res_cpu.flatten()[idx])
    print("tpu:", res_tpu.cpu().flatten()[idx])

def case_eltwise():
    device = "privateuseone"
    N = 1
    C = 512
    H = 64
    W = 64

    inp = torch.randn(N, C, H, W)
    t1 = torch.randn(N, C, H, W)
    t2 = torch.randn(N, C, H, W)
    
    inp_tpu = inp.to(device)
    t1_tpu = t1.to(device)
    t2_tpu = t2.to(device)

    res_cpu = torch.addcmul(inp, t1, t2)
    res_tpu = torch.addcmul(inp_tpu, t1_tpu, t2_tpu)
    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)
    print("cpu:", res_cpu.flatten()[idx])
    print("tpu:", res_tpu.cpu().flatten()[idx])


if __name__ == "__main__":
    #case_eltwise()
    case_broadcast()
