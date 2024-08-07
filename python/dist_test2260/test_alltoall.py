import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
TPU = "tpu"

# get rank and world_size from env
rank = os.environ.get("LOCAL_RANK")
world_size = os.environ.get("LOCAL_WORLD_SIZE")
if rank == None:
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", 0)
    world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", 1)

# test tensor length
tensor_len = 16
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

def case1():
    input_tensor = torch.rand(tensor_len)
    logger.info("rank: {}, input_tensor: {}".format(rank, input_tensor))
    device = torch.device(f"{TPU}:{chip_map[int(rank)]}")
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)

    output_tensor = torch.zeros(tensor_len)
    logger.info("rank: {}, output_tensor: {}".format(rank, output_tensor))
    if torch_tpu.tpu.current_device() == device.index:
        output_tensor = output_tensor.to(device)

    dist.all_to_all_single(output_tensor, input_tensor)

    results = output_tensor.cpu()
    logger.info("rank: {}, result: {}".format(rank, results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_alltoall.py 2>&1 | tee 1.log