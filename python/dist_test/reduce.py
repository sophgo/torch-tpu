import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import torch_tpu
import os
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", 0)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", 1)

torch_tpu.tpu.set_device(int(rank))
dist.init_process_group(backend="scclHost", rank=int(rank), world_size=int(world_size))
init_logger()

if is_master():
    tensor = torch.tensor([3, 4]).to(TPU)

if is_slave():
    tensor = torch.tensor([5, 6]).to(TPU)

logging.info(f"before reduce: {tensor.cpu()}")

dist.reduce(tensor, op=dist.ReduceOp.SUM, dst=0)

logging.info(f"after reduce: {tensor.cpu()}")