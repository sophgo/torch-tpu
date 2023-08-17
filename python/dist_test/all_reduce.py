import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import sccl_collectives
torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
TPU = "privateuseone"

dist.init_process_group(backend="sccl")
init_logger()

if is_master():
    tensor = torch.tensor([3.0, 4.0]).to(TPU)

if is_slave():
    tensor = torch.tensor([5.0, 6.0]).to(TPU)

logging.info(f"before reduce: {tensor.cpu()}")

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

logging.info(f"after reduce: {tensor.cpu()}")

