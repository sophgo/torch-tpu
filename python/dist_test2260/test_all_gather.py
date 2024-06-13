import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import torch_tpu
import sccl
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

tensor_len = 4
# init dist and logger
options = sccl.ProcessGroupSCCLOptions()
chip_map = [0,1,2,3,4,5,6,7]
options.chip_map = chip_map
torch_tpu.tpu.set_device(options.chip_map[int(rank)])
dist.init_process_group(backend="sccl", rank=int(rank), world_size=int(world_size), pg_options=options)
init_logger()

def case1():

    input_tensor = torch.rand(tensor_len)
    logging.info("rank: {}, {}".format(rank, input_tensor))
    device = torch.device(f"{TPU}:{chip_map[int(rank)]}")
    if torch_tpu.tpu.current_device() == device.index:
        input_tensor = input_tensor.to(device)

    output_list = [torch.zeros(tensor_len) for _ in range(int(world_size))]
    logging.info("rank: {}, {}".format(rank, output_list))
    output_list = [tensor.to(device) for tensor in output_list if torch_tpu.tpu.current_device() == device.index]

    dist.all_gather(output_list, input_tensor)

    results = [tensor.cpu() for tensor in output_list]
    logging.info("rank: {}, results: {}".format(rank,results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 8 -output-filename log python test_all_gather.py 2>&1 | tee 1.log