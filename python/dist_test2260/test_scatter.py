import time
import torch
import torch.distributed as dist
import logging
import os
from helper import init_logger, is_master, is_slave
import time
import torch_tpu
import sccl_collectives
TPU = "tpu"

rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
world_size = os.environ.get("OMPI_COMM_WORLD_SIZE", None)

tensor_len = 4

dist.init_process_group(backend="SOPHON", rank=int(rank), world_size=int(world_size))
init_logger()

def case1():
    if is_master():
        input_list = [torch.rand(tensor_len) for _ in range(int(world_size))]
        logging.info("rank: {}, input_list: {}".format(rank, input_list))
        input_list = [tensor.to(TPU) for tensor in input_list]
    
    output_tensor = torch.zeros(tensor_len)
    logging.info("rank: {}, output_tensor: {}".format(rank, output_tensor))
    output_tensor = output_tensor.to(TPU)

    if is_master():
        dist.scatter(output_tensor, input_list, src=0)
    else:
        dist.scatter(output_tensor, src=0)
    
    results = output_tensor.cpu()
    logging.info("rank: {}, results: {}".format(rank, results))


if __name__ == "__main__":
    case1()

# mpirun --allow-run-as-root -n 2 -output-filename log python test_scatter.py 2>&1 | tee 1.log