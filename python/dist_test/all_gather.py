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

tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(2)]

dist.all_gather(tensor_list, tensor)

logging.info(f"gathered: {tensor_list}")