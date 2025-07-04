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
# torchrun --nproc_per_node 2 --nnodes 1 test_all_gather.py
# tensor_len = 65536*76032
# tensor_len = 16080*76032*2
dtype = torch.float16
# tensor_len = 80 * 76032
# tensor_len = 8192
tensor_len = 4
# init dist and logger
options = torch_tpu.ProcessGroupSCCLOptions()
torch_tpu.tpu.set_chip_map(options, use_rank_table=False)
torch_tpu.tpu.set_device(rank)
dist.init_process_group(backend="sccl", rank=rank, world_size=world_size, pg_options=options)
init_logger()
logger = logging.getLogger('sccl_logger')

def case1():

    input_tensor = torch.rand(tensor_len, dtype=dtype)
    logger.info("rank: {}, {}".format(rank, input_tensor))
    device = torch.device(f"{TPU}:{rank}")
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)

    output_list = [torch.zeros(tensor_len, dtype=dtype) for _ in range(int(world_size))]
    logger.info("rank: {}, {}".format(rank, output_list))
    output_list = [tensor.to(device) for tensor in output_list if torch_tpu.tpu.current_device() == device.index]

    dist.all_gather(output_list, input_tensor)

    results = [tensor.cpu() for tensor in output_list]
    logger.info("rank: {}, results: {}".format(rank,results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_all_gather.py 2>&1 | tee 1.log