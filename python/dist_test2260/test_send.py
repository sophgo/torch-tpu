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
chip_map = [0,1,2,3,4,5,6,7]
if torch_tpu.tpu.is_rank_table_valid():
    chip_map = torch_tpu.tpu.read_rank_table()
options.chip_map = chip_map
torch_tpu.tpu.set_device(options.chip_map[int(rank)])
dist.init_process_group(backend="sccl", rank=int(rank), world_size=int(world_size), pg_options=options)
init_logger()
logger = logging.getLogger('sccl_logger')

device = torch.device(f"{TPU}:{chip_map[int(rank)]}")

if int(rank) == 0:
    tensor = torch.rand(tensor_len)
    logger.info(f"{int(rank)} input: {tensor}")
    print("rank:", rank, tensor)
    if torch_tpu.tpu.current_device() == device.index:
        tensor = tensor.to(device)
    dist.isend(tensor, dst=1)
    # req.wait()
    print('Rank 0 has sent the tensor to Rank 1')
elif int(rank) == 1:
    tensor = torch.zeros(tensor_len)
    logger.info(f"{int(rank)} input: {tensor}")
    print("rank:", rank, tensor)
    if torch_tpu.tpu.current_device() == device.index:
        tensor = tensor.to(device)
    dist.irecv(tensor, src=0)
    # req.wait()
    print('Rank 1 has received the tensor:', tensor.cpu())

# mpirun --allow-run-as-root  -n 8 -output-filename log python test_send.py 2>&1 | tee 1.log