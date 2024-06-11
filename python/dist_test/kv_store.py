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

store = dist.TCPStore(
    host_name="127.0.0.1",
    port=5848,
    world_size=dist.get_world_size(),
    is_master=is_master(),
)

if is_slave():
    store.set("the_key", "Hi master")
    logging.info("slave has set the key")

if is_master():
    result = store.get("the_key")
    logging.info(f"master has get the value: {result}")
    
    b64tensor = b64encode(pickle.dumps(torch.rand(4))).decode()
    store.set("another_key", b64tensor)
    logging.info(f"master set a torch.Tensor object to the same key")
    time.sleep(2)  # should wait for the slave to read the data before quit

if is_slave():
    time.sleep(1)  # wait for master to set the torch.Tensor object
    store.get("another_key")
    tensor = pickle.loads(b64decode(store.get("another_key")))
    logging.info(f"slave got object from the same key {tensor}")
    
# how to use TPU?