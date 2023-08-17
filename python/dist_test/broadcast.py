import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave

dist.init_process_group(backend="gloo")
init_logger()

if is_slave():
    tensor = torch.zeros(3)

if is_master():
    tensor = torch.rand(3)

dist.broadcast(tensor, src=0)

time.sleep(1)

logging.info(tensor)
