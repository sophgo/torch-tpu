import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
import sccl_collectives
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

tensor_len = 4

dist.init_process_group(backend="SOPHON", rank=int(rank), world_size=int(world_size))
init_logger()

if is_slave():
    tensor = torch.zeros(tensor_len)
    logging.info(tensor)
    print(tensor)
    tensor = tensor.to(TPU)

if is_master():
    tensor = torch.rand(tensor_len)
    logging.info(tensor)
    print(tensor)
    tensor = tensor.to(TPU)


dist.broadcast(tensor, src=0)

logging.info(tensor.cpu())

# mpirun --allow-run-as-root  -n 2 -output-filename log python test_broadcast.py 2>&1 | tee 1.log
