import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
import sccl_collectives
TPU = "tpu"


rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

tensor_len = 4
torch_tpu.tpu.set_device(int(rank))
dist.init_process_group(backend="SOPHON", rank=int(rank), world_size=int(world_size))
init_logger()

def case1():
    logging.info("rank: {}".format(rank))
    master_results = None
    slave_results = None

    if is_master():
        tensor = torch.rand(tensor_len)
        logging.info("rank: {}, {}".format(rank, tensor))
        tensor = tensor.to(TPU)

    if is_slave():
        tensor = torch.rand(tensor_len)
        logging.info("rank: {}, {}".format(rank, tensor))
        tensor = tensor.to(TPU)

    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)

    if is_master():
        master_results = tensor.cpu()
        logging.info("rank: {}, master_results: {}".format(rank, master_results))

    if is_slave():
        slave_results = tensor.cpu()
        logging.info("rank: {}, slave_results: {}".format(rank, slave_results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_reduce.py 2>&1 | tee 1.log
