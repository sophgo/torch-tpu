import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    dim = None
    input_origin=torch.rand(2,4,2,4,4,6)
    input_tpu=input_origin.to(device)
    output_cpu=torch.argmin(input_origin,dim=dim)
    output_tpu=torch.argmin(input_tpu,dim=dim).cpu()
    # print("input_origin : ",input_origin)
    print("output_cpu : ", output_cpu.shape)
    print("output_tpu : ", output_tpu.shape)
    print("delta : ",(output_cpu==output_tpu).all())

def case2():
    dim = 2
    dtypes = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int16,
        torch.int8
    ]

    for dtype in dtypes:
        print(f"\n=== Testing dtype: {dtype} ===")

        if dtype in [torch.float32, torch.float16, torch.bfloat16]:
            input_origin = torch.rand(256, 2, 256, 256).to(dtype) * 10.0
        else:
            input_origin = torch.randint(-100, 100, (256, 2, 256, 256), dtype=dtype)

        input_tpu = input_origin.to(device)
        output_cpu = torch.argmin(input_origin, dim=dim, keepdim=True)
        output_tpu = torch.argmin(input_tpu, dim=dim, keepdim=True).cpu()

        if dtype in [torch.int32, torch.int16, torch.int8]:
            output_cpu = output_cpu.to(torch.int64)
            output_tpu = output_tpu.to(torch.int64)

        print(f"input_origin shape: {input_origin.shape}, dtype: {input_origin.dtype}")
        print(f"output_cpu shape: {output_cpu.shape}, dtype: {output_cpu.dtype}")
        print(f"output_tpu shape: {output_tpu.shape}, dtype: {output_tpu.dtype}")
        print(f"All elements match: {(output_cpu == output_tpu).all()}")

        print("First 10 elements comparison:")
        cpu_flat = output_cpu.flatten()[:10]
        tpu_flat = output_tpu.flatten()[:10]
        for i, (cpu_val, tpu_val) in enumerate(zip(cpu_flat, tpu_flat)):
            match_flag = "✓" if cpu_val == tpu_val else "✗"
            print(f"  [{i}] CPU: {cpu_val:6d}, TPU: {tpu_val:6d} {match_flag}")

if __name__ == "__main__":
    case1()
    # case2()

