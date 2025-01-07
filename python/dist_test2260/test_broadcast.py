import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
TPU = "tpu"

# get rank and world_size from env
rank = os.environ.get("RANK")
world_size = os.environ.get("WORLD_SIZE")
if rank == None:
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", 0)
    world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", 1)

tensor_len = 4
# init dist and logger
options = torch_tpu.ProcessGroupSCCLOptions()
torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
torch_tpu.tpu.set_device(options.chip_map[int(rank)])
dist.init_process_group(backend="sccl", rank=int(rank), world_size=int(world_size), pg_options=options)
init_logger()
logger = logging.getLogger('sccl_logger')

if is_slave():
    tensor = torch.zeros(tensor_len)
    # tensor = torch.zeros(tensor_len).to(torch.bool)
    logger.info(f"{int(rank)} input: {tensor}")
    print(tensor)
    device = torch.device(f"{TPU}:{options.chip_map[int(rank)]}")
    if torch_tpu.tpu.current_device() == device.index:
        tensor = tensor.to(device)

if is_master():
    tensor = torch.rand(tensor_len)
    # tensor = torch.randint(0, 2, (tensor_len,), dtype=torch.bool)
    logger.info(f"{int(rank)} input: {tensor}")
    print(tensor)
    device = torch.device(f"{TPU}:{options.chip_map[int(rank)]}")
    if torch_tpu.tpu.current_device() == device.index:
        tensor = tensor.to(device)

dist.broadcast(tensor, src=0)

logger.info(f"{int(rank)} result: {tensor.cpu()}")

# mpirun --allow-run-as-root  -n 8 -output-filename log python test_broadcast.py 2>&1 | tee 1.log
