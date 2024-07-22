import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import sccl
TPU = "tpu"

import scclHost
import torch_tpu
import os
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

torch_tpu.tpu.set_device(int(rank))
dist.init_process_group(backend="scclHost", rank=int(rank), world_size=int(world_size))
init_logger()

if is_master():
    tensor = torch.tensor([3.0, 4.0]).to(TPU)

if is_slave():
    tensor = torch.tensor([5.0, 6.0]).to(TPU)

logging.info(f"before reduce: {tensor.cpu()}")

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

logging.info(f"after reduce: {tensor.cpu()}")