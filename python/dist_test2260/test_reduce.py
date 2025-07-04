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
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

tensor_len = 4
# init dist and logger
options = torch_tpu.ProcessGroupSCCLOptions()
torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
torch_tpu.tpu.set_device(rank)
dist.init_process_group(backend="sccl", rank=rank, world_size=world_size, pg_options=options)
init_logger()
logger = logging.getLogger('sccl_logger')

def case1():
    logger.info("rank: {}".format(rank))
    master_results = None
    slave_results = None
    device = torch.device(f"{TPU}:{rank}")

    if is_master():
        tensor = torch.rand(tensor_len)
        logger.info("rank: {}, {}".format(rank, tensor))
        if torch_tpu.tpu.current_device() == device.index:
            tensor = tensor.to(device)

    if is_slave():
        tensor = torch.rand(tensor_len)
        logger.info("rank: {}, {}".format(rank, tensor))
        if torch_tpu.tpu.current_device() == device.index:
            tensor = tensor.to(device)

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    if is_master():
        master_results = tensor.cpu()
        logger.info("rank: {}, master_results: {}".format(rank, master_results))

    if is_slave():
        slave_results = tensor.cpu()
        logger.info("rank: {}, slave_results: {}".format(rank, slave_results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_reduce.py 2>&1 | tee 1.log
