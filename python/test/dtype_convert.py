import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

def uint8_to_bool():
    device = "privateuseone"
    seq = 8
    
    inp = torch.randint(1,1, seq, seq)
    inp_tpu = inp.to(device)

def case_f32_to_f16():
    device = "privateuseone"
    batch = 32
    head_num = 12
    sequence = 1024
    inp = torch.randn(batch, head_num, sequence, sequence)
    inp_tpu = inp.to(device)
    t1 = time.time()
    inp_tpu_half = inp_tpu.half()
    t2 = time.time()
    print("[time]f32->f16:", t2 - t1)

    t1 = time.time()
    inp_tpu_f32 = inp_tpu_half.float()
    t2 = time.time()
    print("[time]f16->f32:", t2 - t1)

def case_int64_to_int32():
    device = "privateuseone"
    batch = 32
    head_num = 12
    sequence = 1024
    inp = torch.randint(1,10000, (batch, head_num, sequence, sequence))
    inp_tpu = inp.to(device)
    inp_tpu_i32 = inp_tpu.int()
    diff = inp - inp_tpu_i32.cpu()
    print("max diff : ",  torch.max(abs(diff)))
    print(diff[:10])

if __name__ == "__main__":
    #uint8_to_bool()
    case_f32_to_f16()
    case_int64_to_int32()