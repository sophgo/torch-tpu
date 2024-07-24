import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave, is_rank_table_valid
import torch_tpu
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", 0)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", 1)

tensor_len = 4
# init dist and logger
options = torch_tpu.ProcessGroupSCCLOptions()
chip_map = [0,1,2,3,4,5,6,7]
if is_rank_table_valid():
    chip_map = torch_tpu.tpu.read_rank_table()
options.chip_map = chip_map
torch_tpu.tpu.set_device(options.chip_map[int(rank)])
dist.init_process_group(backend="sccl", rank=int(rank), world_size=int(world_size), pg_options=options)
init_logger()

def case1():
    logging.info("rank: {}".format(rank))
    master_results = None
    slave_results = None
    device = torch.device(f"{TPU}:{chip_map[int(rank)]}")

    if is_master():
        tensor = torch.rand(tensor_len)
        logging.info("rank: {}, {}".format(rank, tensor))
        if torch_tpu.tpu.current_device() == device.index:
            tensor = tensor.to(device)

    if is_slave():
        tensor = torch.rand(tensor_len)
        logging.info("rank: {}, {}".format(rank, tensor))
        if torch_tpu.tpu.current_device() == device.index:
            tensor = tensor.to(device)

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
