import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
#from utils import compare_model_grad
import torch_tpu
torch.tpu = torch_tpu.tpu

device = 'tpu'

def storage():
    x = torch.rand((10)).tpu()
    x_storage = x.untyped_storage()
    storage_cpu = torch.UntypedStorage(10)
    import pdb;pdb.set_trace()
    storage_cpu.copy_(x_storage ,False)

def tensor():
    x_t = torch.rand((10)).tpu()
    x_c = torch.empty_like(x_t, device='cpu')
    x_c.copy_(x_t, False)
    diff = x_t.cpu() - x_c
    print(torch.max(abs(diff)))

def empty_cache():
    import os
    os.environ["TPU_ALLOCATOR_FREE_DELAY_IN_MS"] = "100000" #100s
    def buffer():
        x = torch.empty((1000,1000), dtype=torch.int8, device=device)
    x_t = torch.randn((1000,1000)).tpu()
    import pdb; pdb.set_trace()
    buffer()
    import pdb; pdb.set_trace()
    torch_tpu.tpu.empty_cache()
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    # storage()
    # tensor()
    empty_cache()