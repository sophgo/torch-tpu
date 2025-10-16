from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
dtype = torch.float16

def case1():
    input_ids = torch.tensor([[2,0,1,1],[5,3,4,3]], device=device, dtype=torch.int32)
    token_freq = torch.zeros((2, 10), device=input_ids.device, dtype=dtype)
    token_freq.scatter_add_(
            1, input_ids, torch.ones_like(input_ids, dtype=dtype)
        )
    print(token_freq.cpu())

def case2():
    # frequency calculation for generated tokens in TGI
    rows = 4
    voc_size = 32000
    input_length = 10
    def check(rows, voc_size, input_length):
        input_ids_cpu = torch.randint(low=0, high=voc_size, size=(rows, input_length), dtype=torch.int32)
        token_freq_cpu = torch.zeros((rows, voc_size), dtype=torch.float)
        input_ids_tpu = input_ids_cpu.to(device)
        token_freq_tpu = token_freq_cpu.to(device)

        token_freq_cpu.scatter_add_(
                1, input_ids_cpu.to(torch.int64), torch.ones_like(input_ids_cpu, dtype=torch.float)
            )
        token_freq_tpu.scatter_add_(
                1, input_ids_tpu, torch.ones_like(input_ids_tpu, dtype=torch.float)
            )
        return torch.max(abs(token_freq_tpu.cpu() - token_freq_cpu)) < 1e-4

    for rows in [1, 4, 7, 13, 16]:
        for voc_size in [32000, 128256, 152064]:
            for input_length in [1, 6, 15, 33, 155, 1022]:
                success = check(rows=rows, voc_size=voc_size, input_length=input_length)
                print(f'CHECKING: {rows=}, {voc_size=}, {input_length=}, Success: {success}')
                if not success:
                    raise "CHECK ERROR!!"

    # for rows in [1, 5, 16]:
    #     for voc_size in [152064]:
    #         for input_length in [155]:
    #             success = check(rows=rows, voc_size=voc_size, input_length=input_length)
    #             print(f'CHECKING: {rows=}, {voc_size=}, {input_length=}, Success: {success}')
    #             if not success:
    #                 raise "CHECK ERROR!!"

def case3():
    dtypes = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int16,
        torch.int8
    ]

    for dtype in dtypes:
        input_ids_cpu = torch.tensor([[2, 0, 1, 1], [5, 3, 4, 3]], dtype=torch.int32)
        token_freq_cpu = torch.zeros((2, 10), dtype=dtype)
        ones_cpu = torch.ones_like(input_ids_cpu, dtype=dtype)

        input_ids = input_ids_cpu.contiguous().to(device)
        token_freq = token_freq_cpu.contiguous().to(device)
        ones = ones_cpu.contiguous().to(device)

        token_freq.scatter_add_(1, input_ids, ones)


        token_freq_cpu.scatter_add_(
                1, input_ids_cpu.to(torch.int64), ones_cpu
            )
        # print("TPU:",token_freq.cpu())
        # print("CPU:",token_freq_cpu)
        print(f"max_diff: {torch.max(abs(token_freq_cpu - token_freq.cpu()))}")

if __name__ == "__main__":
    case1()
    # case3()
