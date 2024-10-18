import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"


def case_add():
    out = []
    print("#"*5 + "case_add" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))

    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    print(out)
    

def case_add_bcast():
    out = []
    print("#"*5 + "case_add_bcast" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a + b
        tpu_out = a.to(device) + b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))

    print(out)

def case_add_scalar():
    out = []
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
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out - tpu_out2.cpu())))
    
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
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out - tpu_out2.cpu())))
    
    print(out)

def case_sub():
    out = []
    print("#"*5 + "case_sub" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a - b
        tpu_out = a.to(device) - b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))

    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a - b
        tpu_out = a.to(device) - b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    print(out)
    

def case_sub_bcast():
    out= []
    print("#"*5 + "case_sub_bcast" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a - b
        tpu_out = a.to(device) - b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    for dtype_t in [torch.int32, torch.int16, torch.int8]:
        a = torch.randint(0, 21, (3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a - b
        tpu_out = a.to(device) - b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    print(out)

def case_sub_scalar():
    out = []
    print("#"*5 + "case_sub_scalar" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = 3.0

        cpu_out = a - b
        cpu_out2 = b - a
        tpu_out = a.to(device) - b
        tpu_out2 = b - a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu1: {cpu_out.shape}, cpu2: {cpu_out2.shape} tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out2 - tpu_out2.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out2 - tpu_out2.cpu())))
    
    for dtype_t in [torch.int32, torch.int16, torch.int8]:
        a = torch.randint(0, 21, (3, 4, 2, 6), dtype=dtype_t)
        b = 3

        cpu_out = a - b
        cpu_out2 = b - a
        tpu_out = a.to(device) - b
        tpu_out2 = b - a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu1: {cpu_out.shape}, cpu2: {cpu_out2.shape} tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out2 - tpu_out2.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out2 - tpu_out2.cpu())))
    
    print(out)

def case_mul():
    out = []
    print("#"*5 + "case_mul" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a * b
        tpu_out = a.to(device) * b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))

    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a * b
        tpu_out = a.to(device) * b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    print(out)
    

def case_mul_bcast():
    out = []
    print("#"*5 + "case_mul_bcast" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a * b
        tpu_out = a.to(device) * b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    for dtype_t in [torch.int32, torch.int16, torch.int8]:
        a = torch.randint(0, 21, (5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(0, 21, (1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a * b
        tpu_out = a.to(device) * b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    print(out)

def case_mul_scalar():
    out = []
    print("#"*5 + "case_mul_scalar" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = 3.0

        cpu_out = a * b
        cpu_out2 = b * a
        tpu_out = a.to(device) * b
        tpu_out2 = b * a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu1: {cpu_out.shape}, cpu2: {cpu_out2.shape} tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out2 - tpu_out2.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out2 - tpu_out2.cpu())))
    
    for dtype_t in [torch.int32, torch.int16, torch.int8]:
        a = torch.randint(0, 21, (3, 4, 2, 6), dtype=dtype_t)
        b = 3

        cpu_out = a * b
        cpu_out2 = b * a
        tpu_out = a.to(device) * b
        tpu_out2 = b * a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu1: {cpu_out.shape}, cpu2: {cpu_out2.shape} tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out2 - tpu_out2.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out2 - tpu_out2.cpu())))
    
    print(out)

def case_div():
    out = []
    print("#"*5 + "case_div" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t) + 0.01

        cpu_out = a / b
        tpu_out = a.to(device) / b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"dtype: {cpu_out.dtype, tpu_out.dtype}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))

    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(1, 21, (5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(1, 21, (5, 3, 4, 2, 6), dtype=dtype_t)

        cpu_out = a / b
        tpu_out = a.to(device) / b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"dtype: {cpu_out.dtype, tpu_out.dtype}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    print(out)

def case_div_bcast():
    out = []
    print("#"*5 + "case_div_bcast" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randn((1, 4, 1, 6), dtype=dtype_t) + 0.01

        cpu_out = a / b
        tpu_out = a.to(device) / b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    for dtype_t in [torch.int32, torch.int16, torch.int8]:
        a = torch.randint(1, 21, (5, 3, 4, 2, 6), dtype=dtype_t)
        b = torch.randint(1, 21, (1, 4, 1, 6), dtype=dtype_t)

        cpu_out = a / b
        tpu_out = a.to(device) / b.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu: {cpu_out.shape}, tpu: {tpu_out.shape}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
    
    print(out)

def case_div_scalar():
    out = []
    print("#"*5 + "case_div_scalar" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((5, 3, 4, 2, 6), dtype=dtype_t) + 0.01
        b = torch.tensor(3.0)

        cpu_out = a / b
        cpu_out2 = b / a
        tpu_out = a.to(device) / b
        tpu_out2 = b.to(device) / a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu1: {cpu_out.shape}, cpu2: {cpu_out2.shape} tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out2 - tpu_out2.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out2 - tpu_out2.cpu())))
    
    for dtype_t in [torch.int32, torch.int16, torch.int8]:
        a = torch.randint(1, 21, (3, 4, 2, 6), dtype=dtype_t)
        b = torch.tensor(3)

        cpu_out = a / b
        cpu_out2 = b / a
        tpu_out = a.to(device) / b
        tpu_out2 = b.to(device) / a.to(device)
        print(f"dtype: {dtype_t}")
        print(f"shape: cpu1: {cpu_out.shape}, cpu2: {cpu_out2.shape} tpu1: {tpu_out.shape}, tpu2: {tpu_out2.shape}")
        print(f"tpu1 max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        print(f"tpu2 max_diff: {torch.max(abs(cpu_out2 - tpu_out2.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))
        out.append(torch.max(abs(cpu_out2 - tpu_out2.cpu())))

    print(out)

def test_add():
    case_add()
    case_add_bcast()
    case_add_scalar()

def test_sub():
    case_sub()
    case_sub_bcast()
    case_sub_scalar()

def test_mul():
    case_mul()
    case_mul_bcast()
    case_mul_scalar()

def test_div():
    case_div()
    case_div_bcast()
    case_div_scalar()

def add_test():
    a = torch.randn((32, 6, 50304))
    b = torch.randn((32, 6, 1))

    cpu_out = a * b
    tpu_out = a.to(device) * b.to(device)

    print(f"shape: cpu_out: {cpu_out.shape}, tpu_out: {tpu_out.shape}")
    print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
    print(f"where: {torch.unique(torch.where(abs(cpu_out - tpu_out.cpu()) > .1)[1])}")


if __name__ == "__main__":
    test_add()
    test_sub()
    test_mul()
    test_div()

    add_test()



    
