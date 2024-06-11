import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
import sccl
TPU = "tpu"

# get rank and world_size from env
rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

# test tensor length
tensor_len = 8
torch_tpu.tpu.set_device(int(rank))
# init dist and logger
options = sccl.ProcessGroupSCCLOptions()
# options.chip_map = [0, 1]
dist.init_process_group(backend="sccl", rank=int(rank), world_size=int(world_size), pg_options=options)
init_logger()

def case1():
    input_tensor = torch.rand(tensor_len)
    logging.info("rank: {}, input_tensor: {}".format(rank, input_tensor))
    input_tensor = input_tensor.to(TPU)

    output_tensor = torch.zeros(tensor_len)
    logging.info("rank: {}, output_tensor: {}".format(rank, output_tensor))
    output_tensor = output_tensor.to(TPU)

    dist.all_to_all_single(output_tensor, input_tensor)

    results = output_tensor.cpu()
    logging.info("rank: {}, result: {}".format(rank, results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_alltoall.py 2>&1 | tee 1.log