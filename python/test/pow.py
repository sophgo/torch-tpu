import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from test_utils import *

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case_pow_c():
    for dtype in [torch.float, torch.float16, torch.bfloat16, torch.int32]:
        if dtype == torch.int32:
            x = torch.randint(-5, 12, (4, 4, 64, 500), dtype=dtype)
        else:
            x = torch.randn((3, 4), dtype=dtype)
        for exponent in [2, 3.5]:

            out_cpu = x ** exponent
            out_tpu = x.to(device) ** exponent

            # nan in tpu is 0
            out_cpu[torch.isnan(out_cpu)] = 0

            print(f"{dtype}_{exponent} fp comparison")
            print(f"shpae, dtype, cpu: {out_cpu.shape, out_cpu.dtype}, tpu: {out_tpu.shape, out_cpu.dtype}")
            # print(f"cpu_fp: {out_cpu}")
            # print(f"tpu_fp: {out_tpu.cpu()}")
            print(f"max diff: {torch.max(torch.abs(out_cpu - out_tpu.cpu()))}")

def case_c_pow():
    for dtype in [torch.float, torch.float16, torch.bfloat16, torch.int32]:
        if dtype == torch.int32:
            exponent = torch.randint(-5, 12, (4, 4, 64, 500), dtype=dtype)
        else :
            exponent = torch.randn((4, 4, 64, 500), dtype=dtype)
        # print(f"exponent: {exponent}")
        for x in [2, 3.5]:

            out_cpu = x ** exponent
            out_tpu = torch.pow(x, exponent.to(device))

            # nan in tpu is 0
            out_cpu[torch.isnan(out_cpu)] = 0

            print(f"{dtype}_{x} fp comparison")
            print(f"shpae, dtype, cpu: {out_cpu.shape, out_cpu.dtype}, tpu: {out_tpu.shape, out_cpu.dtype}")
            # print(f"cpu_fp: {out_cpu_fp}")
            # print(f"tpu_fp: {out_tpu_fp.cpu()}")
            max_diff = torch.max(torch.abs(out_cpu - out_tpu.cpu()))
            print(f"max diff: {max_diff}")

def case_pow():
    for dtype in [torch.float, torch.float16, torch.bfloat16, torch.int32]:
        shape = (4, 5)
        if dtype == torch.int32:
            x = torch.randint(-5, 12, shape, dtype=dtype)
            exponent = torch.randint(0, 5, shape, dtype=dtype)
        else:
            x = torch.rand(shape, dtype=dtype)
            exponent = torch.rand(shape, dtype=dtype)
    
        out_cpu = x ** exponent
        out_tpu = torch.pow(x.to(device), exponent.to(device))

        # nan in tpu is 0
        # out_cpu[torch.isnan(out_cpu)] = 0

        print(f"{dtype} comparison")
        print(f"shpae, dtype, cpu: {out_cpu.shape, out_cpu.dtype}, tpu: {out_tpu.shape, out_cpu.dtype}")
        
        print(f"cpu_fp: {out_cpu}")
        print(f"tpu_fp: {out_tpu.cpu()}")
        max_diff = torch.max(torch.abs(out_cpu - out_tpu.cpu()))
        print(f"max diff: {max_diff}")

def case_pow():
    for dtype in [torch.float, torch.float16, torch.bfloat16, torch.int32]:
        xshape = (64, 64)
        eshape = (64, 64)
        if dtype == torch.int32:
            x = torch.randint(1, 13, xshape, dtype=dtype)
            exponent = torch.randint(0, 5, eshape, dtype=dtype)
        else:
            x = torch.rand(xshape, dtype=dtype)
            exponent = torch.rand(eshape, dtype=dtype)
    
        out_cpu = x ** exponent
        out_tpu = torch.pow(x.to(device), exponent.to(device))

        # nan in tpu is 0
        out_cpu[torch.isnan(out_cpu)] = 0

        print(f"{dtype} comparison")
        print(f"shpae, dtype, cpu: {out_cpu.shape, out_cpu.dtype}, tpu: {out_tpu.shape, out_cpu.dtype}")
        
        # print(f"cpu_fp: {out_cpu}")
        # print(f"tpu_fp: {out_tpu.cpu()}")
        max_diff = torch.max(torch.abs(out_cpu - out_tpu.cpu()))
        print(f"max diff: {max_diff}")

def case_pow_broadcast():
    for dtype in [torch.float, torch.float16, torch.bfloat16, torch.int32]:
        xshape = (64, 2, 4, 3, 6)
        eshape = (4, 1, 6)
        if dtype == torch.int32:
            x = torch.randint(1, 13, xshape, dtype=dtype)
            exponent = torch.randint(0, 5, eshape, dtype=dtype)
        else:
            x = torch.rand(xshape, dtype=dtype)
            exponent = torch.rand(eshape, dtype=dtype)
    
        out_cpu = x ** exponent
        out_tpu = torch.pow(x.to(device), exponent.to(device))

        # nan in tpu is 0
        out_cpu[torch.isnan(out_cpu)] = 0

        print(f"{dtype} comparison")
        print(f"shpae, dtype, cpu: {out_cpu.shape, out_cpu.dtype}, tpu: {out_tpu.shape, out_cpu.dtype}")
        
        # print(f"cpu_fp: {out_cpu}")
        # print(f"tpu_fp: {out_tpu.cpu()}")
        max_diff = torch.max(torch.abs(out_cpu - out_tpu.cpu()))
        print(f"max diff: {max_diff}")


def temp():
    # a = torch.randn((3, 4), dtype=torch.float)
    # a = torch.arange(1, 13, dtype=torch.float).view(3, 4)
    a = torch.randint(1, 13, (2, 4, 3, 5), dtype=torch.float)
    # b = torch.randint(0, 2, (4, ), dtype=torch.float)
    # b = torch.tensor(2, dtype=torch.int32)
    b = torch.randint(0, 2, (4, 1, 5), dtype=torch.float)

    print(a.dim())
    print(b.dim())
    

    # out_cpu = a ** b
    # out_tpu = a.to(device) ** b.to(device)
    out_cpu = a ** b
    out_tpu = a.to(device) ** b.to(device)
    # nan in tpu is 0
    # out_cpu[torch.isnan(out_cpu)] = 0

    print(f"cpu: {out_cpu}")
    print(f"tpu: {out_tpu.cpu()}")
    max_diff = torch.max(torch.abs(out_cpu - out_tpu.cpu()))
    print(f"max diff: {max_diff}")


if __name__ == "__main__":
    # case_pow_c()
    # case_c_pow()
    # case_pow()
    # case_pow_broadcast()

