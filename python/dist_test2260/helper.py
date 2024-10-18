import torch.distributed as dist
import logging
import os


def init_logger():
    assert dist.is_initialized()
    rank = dist.get_rank()
    role = "master" if rank == 0 else "slave"

    sccl_logger = logging.getLogger('sccl_logger')
    sccl_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("./pytorch-dist.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(f'{role.ljust(6)} ==> %(asctime)8s %(levelname)5s %(message)4s'))
    sccl_logger.addHandler(file_handler)


def is_master():
    return dist.get_rank() == 0


def is_slave():
    return dist.get_rank() != 0