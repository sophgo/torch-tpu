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

if is_slave():
    tensor = torch.zeros(3).to(TPU)

if is_master():
    tensor = torch.rand(3).to(TPU)

dist.broadcast(tensor, src=0)

logging.info(tensor.cpu())
