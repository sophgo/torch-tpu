import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
import sccl_collectives
TPU = "tpu"

# get rank and world_size from env
rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

# test tensor length
tensor_len = 4
torch_tpu.tpu.set_device(int(rank))
# init dist and logger
options = sccl_collectives.ProcessGroupSCCLOptions()
dist.init_process_group(backend="sccl", rank=int(rank), world_size=int(world_size), pg_options=options)
init_logger()

def case1():
    logging.info("rank: {}".format(rank))

    # init tensor
    tensor = torch.rand(tensor_len).float()
    logging.info("rank: {}, {}".format(rank, tensor))
    tensor = tensor.to(TPU)

    # all_reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # print results
    results = tensor.cpu()
    logging.info("rank: {}, results: {}".format(rank, results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_all_reduce.py 2>&1 | tee 1.log
