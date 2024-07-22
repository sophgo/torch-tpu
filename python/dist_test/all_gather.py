import torch
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import scclHost
import torch_tpu
import os
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

torch_tpu.tpu.set_device(int(rank))
dist.init_process_group(backend="scclHost", rank=int(rank), world_size=int(world_size))
init_logger()

if is_master():
    tensor = torch.tensor([3, 4]).to(TPU)

if is_slave():
    tensor = torch.tensor([5, 6]).to(TPU)

tensor_list = [torch.zeros(2, dtype=torch.int64).to(TPU) for _ in range(int(world_size))]

dist.all_gather(tensor_list, tensor)

results = [tensor.cpu() for tensor in tensor_list]
logging.info("rank: {}, results: {}".format(rank, results))