import time
import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import torch_tpu
import os
TPU = "tpu"

# get rank and world_size from env
rank = os.environ.get("LOCAL_RANK")
world_size = os.environ.get("LOCAL_WORLD_SIZE")
if rank == None:
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", 0)
    world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", 1)

torch_tpu.tpu.set_device(int(rank))
dist.init_process_group(backend="scclHost", rank=int(rank), world_size=int(world_size))
init_logger()
logger = logging.getLogger('sccl_logger')

tensor_len = 16
input_tensor = torch.rand(tensor_len)
logger.info("rank: {}, input_tensor: {}".format(rank, input_tensor))
device = torch.device(f"{TPU}:{int(rank)}")
if torch_tpu.tpu.current_device() == device.index:
    input_tensor = input_tensor.to(device)

output_tensor = torch.zeros(tensor_len)
logger.info("rank: {}, output_tensor: {}".format(rank, output_tensor))
if torch_tpu.tpu.current_device() == device.index:
    output_tensor = output_tensor.to(device)

dist.all_to_all_single(output_tensor, input_tensor)

results = output_tensor.cpu()
logger.info("rank: {}, result: {}".format(rank, results))