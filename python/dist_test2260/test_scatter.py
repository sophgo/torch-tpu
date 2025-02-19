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

tensor_len = 16
# init dist and logger
options = torch_tpu.ProcessGroupSCCLOptions()
torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
torch_tpu.tpu.set_device(options.chip_map[rank])
dist.init_process_group(backend="sccl", rank=rank, world_size=world_size, pg_options=options)
init_logger()
logger = logging.getLogger('sccl_logger')

def case1():
    device = torch.device(f"{TPU}:{options.chip_map[rank]}")

    if is_master():
        input_list = [torch.rand(tensor_len) for _ in range(int(world_size))]
        logger.info("rank: {}, input_list: {}".format(rank, input_list))
        input_list = [tensor.to(device) for tensor in input_list if torch_tpu.tpu.current_device() == device.index]

    output_tensor = torch.zeros(tensor_len)
    logger.info("rank: {}, output_tensor: {}".format(rank, output_tensor))
    if torch_tpu.tpu.current_device() == device.index:
        output_tensor = output_tensor.to(device)

    if is_master():
        dist.scatter(output_tensor, input_list, src=0)
    else:
        dist.scatter(output_tensor, src=0)
    results = output_tensor.cpu()
    logger.info("rank: {}, results: {}".format(rank, results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_scatter.py 2>&1 | tee 1.log