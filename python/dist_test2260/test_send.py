import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
TPU = "tpu"

# get rank and world_size from env
rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
if rank == None:
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", 0)
    world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", 1)

tensor_len = 4
# init dist and logger
options = torch_tpu.ProcessGroupSCCLOptions()
torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
torch_tpu.tpu.set_device(options.chip_map[rank])
dist.init_process_group(backend="sccl", rank=rank, world_size=world_size, pg_options=options)
init_logger()
logger = logging.getLogger('sccl_logger')

device = torch.device(f"{TPU}:{options.chip_map[rank]}")

if rank == 0:
    tensor = torch.rand(tensor_len)
    logger.info(f"{rank} input: {tensor}")
    print("rank:", rank, tensor)
    if torch_tpu.tpu.current_device() == device.index:
        tensor = tensor.to(device)
    dist.isend(tensor, dst=1)
    # req.wait()
    print('Rank 0 has sent the tensor to Rank 1')
elif rank == 1:
    tensor = torch.zeros(tensor_len)
    logger.info(f"{rank} input: {tensor}")
    print("rank:", rank, tensor)
    if torch_tpu.tpu.current_device() == device.index:
        tensor = tensor.to(device)
    dist.irecv(tensor, src=0)
    # req.wait()
    print('Rank 1 has received the tensor:', tensor.cpu())

# mpirun --allow-run-as-root  -n 8 -output-filename log python test_send.py 2>&1 | tee 1.log