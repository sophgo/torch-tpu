import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave, is_rank_table_valid
import time
import torch_tpu
import sccl
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

tensor_len = 4
# init dist and logger
options = sccl.ProcessGroupSCCLOptions()
chip_map = [0,1,2,3,4,5,6,7]
if is_rank_table_valid():
    chip_map = torch_tpu.tpu.read_rank_table()
options.chip_map = chip_map
torch_tpu.tpu.set_device(options.chip_map[int(rank)])
dist.init_process_group(backend="sccl", rank=int(rank), world_size=int(world_size), pg_options=options)
init_logger()

def case1():
    master_results = None
    slave_results = None

    input_tensor = torch.rand(tensor_len)
    logging.info("rank: {}, {}".format(rank, input_tensor))
    device = torch.device(f"{TPU}:{chip_map[int(rank)]}")
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)

    if is_master():
        output_list = [torch.zeros(tensor_len) for _ in range(int(world_size))]
        logging.info("rank: {}, {}".format(rank, output_list))
        output_list = [tensor.to(device) for tensor in output_list if torch_tpu.tpu.current_device() == device.index]

    if is_master():
        dist.gather(input_tensor, output_list, dst=0)

    else:
        dist.gather(input_tensor, dst=0)

    if is_master():
        master_results = [tensor.cpu() for tensor in output_list]
        logging.info("rank: {}, master_results: {}".format(rank, master_results))
    
    if is_slave():
        slave_results = input_tensor.cpu()
        logging.info("rank: {}, slave_results: {}".format(rank, slave_results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_gather.py 2>&1 | tee 1.log