import torch
import pickle
import time
from base64 import b64encode, b64decode
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
import sccl
import torch_tpu
TPU = "tpu"

dist.init_process_group(backend="sccl")
init_logger()

if is_master():
    res = torch.rand(4).to(TPU)
    dist.recv(res)
    logging.info(f"master has recv a tensor: {res.cpu()}")
    #time.sleep(2)

if is_slave():
    payload = torch.rand(4).to(TPU)
    #time.sleep(1)  # wait until the master is ready
    res = dist.send(payload, 0)
    logging.info(f"slave has sent a tensor: {payload.cpu()}")
