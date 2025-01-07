import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
TPU = "tpu"

# get rank and world_size from env
rank = int(os.environ.get("RANK"))
world_size = int(os.environ.get("WORLD_SIZE"))
if rank == None:
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))

dtype = torch.float16
tensor_len = 3
# init dist and logger
options = torch_tpu.ProcessGroupSCCLOptions()
torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
torch_tpu.tpu.set_device(options.chip_map[rank])
device = torch.device(f"{TPU}:{options.chip_map[rank]}")

dist.init_process_group(
    backend="sccl",
    rank=rank,
    world_size=world_size,
    pg_options=options)

init_logger()
logger = logging.getLogger('sccl_logger')

def case1():
    input_tensor = torch.rand(tensor_len, dtype=dtype)
    logger.info("rank: {}, input_tensor: {}".format(rank, input_tensor))
    
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)

    # Output in concatenation form
    if rank == 0:
        logger.info("==== Output in concatenation form ====")
    output_tensor = torch.zeros(tensor_len * world_size, dtype=dtype)
    logger.info("rank: {}, output_tensor: {}".format(rank, output_tensor))
    output_tensor = output_tensor.to(device)
    dist.all_gather_into_tensor(output_tensor, input_tensor)
    results = output_tensor.cpu()
    logger.info("rank: {}, results: {}".format(rank, results))

    # Output in stack form
    if rank == 0:
        logger.info("==== Output in stack form ====")
    output_tensor2 = torch.zeros(world_size, tensor_len, dtype=dtype)
    logger.info("rank: {}, output_tensor2: {}".format(rank, output_tensor2))
    output_tensor2 = output_tensor2.to(device)
    dist.all_gather_into_tensor(output_tensor2, input_tensor)

    results2 = output_tensor2.cpu()
    logger.info("rank: {}, results2: {}".format(rank, results2))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output_tensor-filename log python test_all_gather_into_tensor.py 2>&1 | tee 1.log