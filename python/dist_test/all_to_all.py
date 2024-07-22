import time
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

tensor_len = 16
input_tensor = torch.rand(tensor_len)
logging.info("rank: {}, input_tensor: {}".format(rank, input_tensor))
device = torch.device(f"{TPU}:{int(rank)}")
if torch_tpu.tpu.current_device() == device.index:
    input_tensor = input_tensor.to(device)

output_tensor = torch.zeros(tensor_len)
logging.info("rank: {}, output_tensor: {}".format(rank, output_tensor))
if torch_tpu.tpu.current_device() == device.index:
    output_tensor = output_tensor.to(device)

dist.all_to_all_single(output_tensor, input_tensor)

results = output_tensor.cpu()
logging.info("rank: {}, result: {}".format(rank, results))