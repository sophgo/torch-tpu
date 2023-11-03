import torch
import logging
from helper import init_logger, is_master, is_slave
import sccl_collectives
torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
TPU = "privateuseone"

torch.distributed.init_process_group(backend="sccl")

init_logger()
logging.info(f"backend: {torch.distributed.get_backend()}")
logging.info(f"rank: {torch.distributed.get_rank()}")
logging.info(f"world size: {torch.distributed.get_world_size()}")