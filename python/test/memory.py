import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

torch.tpu.empty_cache()

def print_mem_stats():
    a = torch.tpu.memory_stats()
    for k, v in a.items():
        print(f"{k}: {v/1024/1024/1024:.2f} GB")

def print_mem_info():
    free, total = torch.tpu.mem_get_info()
    print(f"Total: {total/1024/1024/1024:.2f} GB, Free: {free/1024/1024/1024:.2f} GB")

print(f'================== Init Stats ==================')
print_mem_stats()
print_mem_info()

torch.tpu.reset_peak_memory_stats()
print(f'================== Reset Peak ==================')
print_mem_stats()
print_mem_info()

all_size = int(1.38 * 1024 * 1024 * 1024)
a = torch.empty((all_size), dtype=torch.uint8, device=device)
print(f'================== alloc {all_size / 1024/1024/1024} ==================')
print_mem_stats()
print_mem_info()

del a
# torch.tpu.reset_peak_memory_stats()
torch.tpu.empty_cache()
torch.tpu.synchronize()
print(f'================== Reset Peak ==================')
print_mem_stats()
print_mem_info()

all_size = int(1.18 * 1024 * 1024 * 1024)
a = torch.empty((all_size), dtype=torch.uint8, device=device)
print(f'================== alloc {all_size / 1024/1024/1024} ==================')
print_mem_stats()
print_mem_info()


all_size = int(0.83 * 1024 * 1024 * 1024)
b = torch.empty((all_size), dtype=torch.uint8, device=device)
print(f'================== alloc {all_size / 1024/1024/1024} ==================')
print_mem_stats()
print_mem_info()

del b
torch.tpu.empty_cache()
torch.tpu.synchronize()
print(f'================== release {all_size / 1024/1024/1024} ==================')

print_mem_stats()
print_mem_info()