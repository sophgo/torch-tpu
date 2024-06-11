import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import sccl
import torch_tpu
TPU = "tpu"

dist.init_process_group(backend="sccl")
init_logger()

if is_slave():
    tensor = torch.zeros(3).to(TPU)

if is_master():
    tensor = torch.rand(3).to(TPU)

dist.broadcast(tensor, src=0)

# time.sleep(1)

logging.info(tensor.cpu())
