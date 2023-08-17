import os

import torch
import sccl_collectives
import torch.distributed as dist
torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
TPU = "privateuseone"

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("sccl", rank=0, world_size=1)

x = torch.ones(6).to(TPU)
dist.all_reduce(x)
print(f"cpu allreduce: {x.cpu()}")

