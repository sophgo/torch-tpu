import torch
import torch.nn.functional as F

def scaled_mm_tpu(x, y, scale_a, scale_b):
    import torch_tpu
    device = "tpu:0"
    x = x.to(device)
    y = y.t().to(device) # transpose y for TPU
    scale_a = scale_a.to(device)
    scale_b = scale_b.to(device)
    import time
    # print("======start timing for running on tpu======")
    start_time = time.time()
    n_loop = 1
    for i in range(n_loop):
        output_tpu = torch._scaled_mm(x, y, scale_a = scale_a, scale_b = scale_b, out_dtype=torch.float32)
        # output_tpu = torch._scaled_mm(x, y)
    torch_tpu.tpu.synchronize()
    end_time = time.time()
    # print(f"x shape: {x.shape}; y shape: {y.shape}; ")
    # print(f"total tpu running time for {n_loop} times: {(end_time - start_time)*1e6:.6f} us;\naverage time for each MM: {((end_time - start_time) / n_loop)*1e6:.6f} us\n")
    print(f"{x.shape[0]} {x.shape[1]} {y.shape[1]} {((end_time - start_time) / n_loop)*1e6:.6f}")
    return output_tpu

def scaled_mm_cpu(x, y, scale_a, scale_b):
    device = "cpu"
    x = x.to(device)
    y = y.to(device)
    x_fp32 = x.to(torch.float32)
    y_fp32 = y.to(torch.float32)
    scale_a = scale_a.to(device)
    scale_b = scale_b.to(device)
    out_fp32 = torch.matmul(x_fp32, y_fp32)
    out_fp32 = out_fp32 / (scale_a * scale_b)
    amax_fp32 = torch.max(torch.abs(out_fp32))
    out_fp8 = out_fp32.to(torch.float8_e4m3fn)
    return out_fp32, out_fp8

def check_scaled_mm(m, k, n):
    torch.manual_seed(1000)
    torch.set_printoptions(precision=6)
    x = (torch.rand((m, k), dtype=torch.float32) * 2 - 1).to(torch.float8_e4m3fn) # [-1, 1]
    y = (torch.rand((k, n), dtype=torch.float32) * 2 - 1).to(torch.float8_e4m3fn) # [-1, 1]
    scale_a = torch.tensor(1)
    scale_b = torch.tensor(1)
    # print(f"x:{x}")
    # print("======cpu computing fp8 mm======")
    out_fp32_cpu, out_fp8_cpu = scaled_mm_cpu(x, y, scale_a, scale_b)
    out_tpu = scaled_mm_tpu(x, y, scale_a, scale_b)[0].cpu()
    print(f"x shape: {x.shape}; y shape: {y.shape}; out_dtype = {out_tpu.dtype}")

    # print("======finish computing, start computing result diff======")
    trans_out_fp32_cpu = out_fp8_cpu.to(torch.float32)
    # print("======cpu out dtype tran precision: ======")
    # print(f"min : {F.cosine_similarity(out_fp32_cpu, trans_out_fp32_cpu).min()}")
    # print(f"max diff: {torch.max(torch.abs(out_fp32_cpu - trans_out_fp32_cpu))}")

    if out_tpu.dtype != torch.float32:
        out_tpu = out_tpu.to(torch.float32)
    diff = torch.abs(out_fp32_cpu - out_tpu)
    cos_simi = F.cosine_similarity(out_fp32_cpu, out_tpu, dim = 1)
    flattened_simi =  F.cosine_similarity(out_fp32_cpu.view(1, -1), out_tpu.view(1, -1), dim = 1)
    # print(f"flattened cos simi: {flattened_simi}")
    if cos_simi.min() >= 0.99:
        print(f"Comparison pass!! Max diff: {diff.max()}; min cos simi: {cos_simi.min()}")
        # import pdb; pdb.set_trace()
    else:
        print(f"Comparison failed!! Max diff: {diff.max()}; min cos simi: {cos_simi.min()}")
        # import pdb; pdb.set_trace()

if __name__ == "__main__":
    # sizes = [1024 * (2**i) for i in range(8)]
    # shape_combinations = [(m, k, n) for m in sizes for k in sizes[0:4] for n in sizes]
    # shape_combinations = [
    #     (2048, 2048, 65536*2),
    #     (2048, 2048*2, 65536*2),
    #     (2048, 2048*4, 65536*2),
    # ]
    shape_combinations = [(500, 4096, 8192)]
    for m, k, n in shape_combinations:
        check_scaled_mm(m, k, n)
