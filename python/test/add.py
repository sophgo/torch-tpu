import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


def case_add():
    print("#"*5 + "case_add" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")

    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
    

def case_add_bcast():
    print("#"*5 + "case_add_bcast" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
    
    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")

def case_add_scalar():
    print("#"*5 + "case_add_scalar" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = 3.0

        cpu_out = a + b
        tpu_out = a.to(device) + b
        tpu_out2 = b + a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out - tpu_out2.cpu()))}")
    
    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (3, 4, 2, 6), dtype=dtype_t)
        b = 3

        cpu_out = a + b
        tpu_out = a.to(device) + b
        tpu_out2 = b + a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out - tpu_out2.cpu()))}")

if __name__ == "__main__":
    # case_add()
    # case_add_bcast()
    case_add_scalar()
