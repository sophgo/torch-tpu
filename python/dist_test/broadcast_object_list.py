import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import sccl_collectives
torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
TPU = "privateuseone"

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