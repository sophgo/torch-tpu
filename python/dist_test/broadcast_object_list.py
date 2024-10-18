import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import torch_tpu
TPU = "tpu"

dist.init_process_group(backend="sccl")
init_logger()

if is_slave():
    objects = [None, None, None]

if is_master():
    objects = ["foo", 12, {1: 2}]

# will get error: int64 is not supported
dist.broadcast_object_list(objects, src=0, device=TPU)

# time.sleep(1)

logging.info(objects)