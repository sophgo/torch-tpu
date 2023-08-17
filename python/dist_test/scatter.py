import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave

dist.init_process_group(backend="gloo")
init_logger()
tensor = torch.tensor([0, 0]).float()

if is_master():
    tensors = [torch.ones(2), torch.ones(2) * 2]
else:
    tensors = None
dist.scatter(tensor, tensors, src=0)

logging.info(f"scattered: {tensor}")