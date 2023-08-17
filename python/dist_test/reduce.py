import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave

dist.init_process_group(backend="gloo")
init_logger()

if is_master():
    tensor = torch.tensor([3, 4])

if is_slave():
    tensor = torch.tensor([5, 6])

logging.info(f"before reduce: {tensor}")

dist.reduce(tensor, op=dist.ReduceOp.SUM, dst=0)

logging.info(f"after reduce: {tensor}")