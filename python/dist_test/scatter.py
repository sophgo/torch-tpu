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
device = torch.device(f"{TPU}:{int(rank)}")
init_logger()
logger = logging.getLogger('sccl_logger')

tensor_len = 16
if is_master():
    input_list = [torch.rand(tensor_len) for _ in range(int(world_size))]
    logger.info("rank: {}, input_list: {}".format(rank, input_list))
    input_list = [tensor.to(device) for tensor in input_list if torch_tpu.tpu.current_device() == device.index]

output_tensor = torch.zeros(tensor_len)
if torch_tpu.tpu.current_device() == device.index:
    output_tensor = output_tensor.to(device)

if is_master():
    dist.scatter(output_tensor, input_list, src=0)
else:
    dist.scatter(output_tensor, src=0)
results = output_tensor.cpu()
logger.info("rank: {}, results: {}".format(rank, results))