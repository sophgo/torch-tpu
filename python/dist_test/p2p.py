import torch
import pickle
import time
from base64 import b64encode, b64decode
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave
torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = "privateuseone"

dist.init_process_group(backend="gloo")
init_logger()

if is_master():
    res = torch.rand(4)
    dist.recv(res)
    logging.info(f"master has recv a tensor: {res}")
    #time.sleep(2)

if is_slave():
    payload = torch.rand(4)
    #time.sleep(1)  # wait until the master is ready
    res = dist.send(payload, 0)
    logging.info(f"slave has sent a tensor: {payload}")

