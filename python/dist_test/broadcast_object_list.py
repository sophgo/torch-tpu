import time
import torch.distributed as dist
import logging
from helper import init_logger, is_master, is_slave

dist.init_process_group(backend="gloo")
init_logger()

if is_slave():
    objects = [None, None, None]

if is_master():
    objects = ["foo", 12, {1: 2}]

dist.broadcast_object_list(objects, src=0)

time.sleep(1)

logging.info(objects)