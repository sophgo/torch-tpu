import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import sccl_collectives
import torch_tpu
TPU = "tpu"

dist.init_process_group(backend="sccl")
init_logger()
tensor = torch.tensor([0, 0]).float().to(TPU)

if is_master():
    tensors = [torch.ones(2).to(TPU), torch.ones(2).to(TPU) * 2]
else:
    tensors = None
dist.scatter(tensor, tensors, src=0)

logging.info(f"scattered: {tensor.cpu()}")