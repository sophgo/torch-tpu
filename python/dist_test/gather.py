import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import sccl_collectives
import torch_tpu
TPU = "tpu"

dist.init_process_group(backend="sccl")
init_logger()

if is_master():
    tensor = torch.tensor([3, 4]).to(TPU)

if is_slave():
    tensor = torch.tensor([5, 6]).to(TPU)

tensor_list = [torch.zeros(2, dtype=torch.int64).to(TPU) for _ in range(2)]

if is_master():
    dist.gather(tensor, tensor_list, dst=0)
else:
    dist.gather(tensor, dst=0)

logging.info(f"gathered: {tensor_list}")