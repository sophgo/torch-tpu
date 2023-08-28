import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)

def case1():
    dtype = torch.float32
    tgt_len = 77
    
    de = "cpu"
    mask_cpu = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=de), device=de)
    mask_cond = torch.arange(mask_cpu.size(-1), device=de)
    s1 = (mask_cond + 1).view(mask_cpu.size(-1),1)
    s2 = mask_cond < s1
    mask_cpu.masked_fill_(s2, 0)
    # mask_cpu = mask_cpu.to(dtype)
    # import pdb;pdb.set_trace()

    TPU = "privateuseone:0"
    mask_tpu = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=TPU), device=TPU)
    mask_cond_tpu = torch.arange(mask_tpu.size(-1), device=TPU)
    s1_t = (mask_cond_tpu + 1).view(mask_tpu.size(-1),1)
    s2_t = mask_cond_tpu < s1_t
    mask_tpu.masked_fill_(s2_t, 0)
    # mask_tpu.masked_fill_(mask_cond < (mask_cond + 1).view(mask_tpu.size(-1), 1), 0)
    # mask_tpu = mask_tpu.to(dtype)

    import pdb;pdb.set_trace()
    diff = mask_cpu - mask_tpu.cpu()
    print("max diff:", torch.max(diff))
    print(diff.flatten()[:10])




    
if __name__ == "__main__":
    case1()