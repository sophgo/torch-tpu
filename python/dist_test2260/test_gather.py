import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import time
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
torch_tpu.tpu.set_device(options.chip_map[rank])
dist.init_process_group(backend="sccl", rank=rank, world_size=world_size, pg_options=options)
init_logger()
logger = logging.getLogger('sccl_logger')

def case1():
    master_results = None
    slave_results = None

    input_tensor = torch.rand(tensor_len)
    logger.info("rank: {}, {}".format(rank, input_tensor))
    device = torch.device(f"{TPU}:{options.chip_map[rank]}")
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)

    if is_master():
        output_list = [torch.zeros(tensor_len) for _ in range(int(world_size))]
        logger.info("rank: {}, {}".format(rank, output_list))
        output_list = [tensor.to(device) for tensor in output_list if torch_tpu.tpu.current_device() == device.index]

    if is_master():
        dist.gather(input_tensor, output_list, dst=0)

    else:
        dist.gather(input_tensor, dst=0)

    if is_master():
        master_results = [tensor.cpu() for tensor in output_list]
        logger.info("rank: {}, master_results: {}".format(rank, master_results))
    
    if is_slave():
        slave_results = input_tensor.cpu()
        logger.info("rank: {}, slave_results: {}".format(rank, slave_results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_gather.py 2>&1 | tee 1.log