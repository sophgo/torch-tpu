import torch.distributed as dist
import logging
import os


def init_logger():
    assert dist.is_initialized()
    rank = dist.get_rank()
    role = "master" if rank == 0 else "slave"

    logging.basicConfig(filename="./pytorch-dist.log",
                        filemode='a',
                        format=f'{role.ljust(6)} ==> %(asctime)8s %(levelname)5s %(message)4s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)


def is_master():
    return dist.get_rank() == 0


def is_slave():
    return dist.get_rank() != 0

def is_rank_table_valid():
    return os.environ.get("RANK_TABLE_FILE", None) is not None