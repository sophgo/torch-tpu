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
    tensor = torch.tensor([3., 4.]).to(TPU)

if is_slave():
    tensor = torch.tensor([5., 6.]).to(TPU)

output_tensor = torch.zeros(2).to(TPU)

logging.info(f"before alltoall: {tensor.cpu()} {output_tensor.cpu()}")
dist.all_to_all_single(output_tensor, tensor)
logging.info(f"after alltoall: {tensor.cpu()} {output_tensor.cpu()}")

