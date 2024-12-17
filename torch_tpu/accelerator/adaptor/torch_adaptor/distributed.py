from functools import wraps
import torch
import torch_tpu
from torch import distributed as dist
from torch.distributed.constants import default_pg_timeout
import os
import time
from nnmoduletools.module_debugger.utils import print_log
from nnmoduletools.module_debugger.tensor_utils import get_tensor_info

class Noop:
    def wait(self):
        return None

def send_recv_wrapper(func):
    @wraps(func)
    def wrapper(tensor, src_dst, group=None, tag=0):
        if dist.get_world_size(group=group) == 1:
            return Noop()
        print_log(f"- {dist.get_backend()}: {func.__name__} {get_tensor_info(tensor)}")
        return func(tensor, src_dst, group=group, tag=tag)
    return wrapper
    
def broadcast_wrapper(func):
    @wraps(func)
    def wrapper(tensor, src, group=None, async_op=False):
        if dist.get_world_size(group=group) == 1:
            return Noop()
        print_log(f"- {dist.get_backend()}: {func.__name__} {get_tensor_info(tensor)}")
        return func(tensor, src, group=group, async_op=async_op)
    return wrapper

def reduce_wrapper(func):
    @wraps(func)
    def wrapper(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if dist.get_world_size(group=group) == 1:
            return Noop()
        print_log(f"- {dist.get_backend()}: {func.__name__} {get_tensor_info(tensor)}")
        if tensor.dtype in [torch.bool, torch.int8, torch.uint8, torch.long]:
            _tensor = tensor.int()
            ret = func(tensor=_tensor, dst=dst, op=op, group=group, async_op=async_op)
            tensor.copy_(_tensor.to(tensor.dtype))
            return ret
        if tensor.dtype == torch.double:
            _tensor = tensor.float()
            ret = func(tensor=_tensor, dst=dst, op=op, group=group, async_op=async_op)
            tensor.copy_(_tensor.to(tensor.dtype))
            return ret
        return func(tensor, dst, op=op, group=group, async_op=async_op)
    return wrapper

def all_reduce_wrapper(func):
    @wraps(func)
    def wrapper(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if dist.get_world_size(group=group) == 1:
            return Noop()
        print_log(f"- {dist.get_backend()}: {func.__name__} {get_tensor_info(tensor)}")
        if tensor.dtype in [torch.bool, torch.int8, torch.uint8, torch.long]:
            _tensor = tensor.int()
            ret = func(tensor=_tensor, op=op, group=group, async_op=async_op)
            tensor.copy_(_tensor.to(tensor.dtype))
            return ret
        if tensor.dtype == torch.double:
            _tensor = tensor.float()
            ret = func(tensor=_tensor, op=op, group=group, async_op=async_op)
            tensor.copy_(_tensor.to(tensor.dtype))
            return ret
        return func(tensor, op=op, group=group, async_op=async_op)
    return wrapper

def all_gather_into_tensor_using_all_gather(output_tensor, input_tensor, group=None, async_op=False):
    print_log(f"- {dist.get_backend()} {get_tensor_info(input_tensor)}: all_gather_into_tensor is not supported. Using All Gather instead.")
    group_size = dist.get_world_size(group=group)
    if group_size == 1:
        output_tensor.copy_(input_tensor)
        return Noop()
    tensor_list = [torch.zeros_like(input_tensor) for _ in range(group_size)]
    handle = dist.all_gather(tensor_list=tensor_list, tensor=input_tensor, group=group, async_op=async_op)
    output_tensor.copy_(torch.cat(tensor_list, dim=0))
    return handle

def init_wrapper(func):
    @wraps(func)
    def wrapper(backend=None, init_method=None, timeout=default_pg_timeout, world_size=-1, rank=-1, store=None, group_name="", pg_options=None):
        if pg_options is None:
            pg_options = torch_tpu.ProcessGroupSCCLOptions()
        if world_size == -1:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        pg_options.chip_map = list(range(world_size))
        if rank == -1:
            rank = int(os.environ.get('RANK', '0'))
        torch_tpu.tpu.set_device(pg_options.chip_map[rank])
        return func(backend=backend, init_method=init_method, timeout=timeout, world_size=world_size, rank=rank, store=store, group_name=group_name, pg_options=pg_options)
    return wrapper

def new_group_wrapper(func):
    @wraps(func)
    def wrapper(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None, use_local_synchronization=False):
        if len(ranks) == 1:
            return func(ranks=ranks, timeout=timeout, backend="gloo", pg_options=pg_options, use_local_synchronization=use_local_synchronization)
        if pg_options is None:
            pg_options = torch_tpu.ProcessGroupSCCLOptions()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        pg_options.chip_map = list(range(world_size))
        rank = os.environ.get('RANK', '0')
        torch_tpu.tpu.set_device(pg_options.chip_map[int(rank)])
        return func(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options, use_local_synchronization=use_local_synchronization)
    return wrapper

    
torch.distributed.broadcast = broadcast_wrapper(torch.distributed.broadcast)
torch.distributed.reduce = reduce_wrapper(torch.distributed.reduce)
torch.distributed.all_reduce = all_reduce_wrapper(torch.distributed.all_reduce)
torch.distributed.all_gather_into_tensor = all_gather_into_tensor_using_all_gather
torch.distributed.distributed_c10d.all_gather_into_tensor = all_gather_into_tensor_using_all_gather
torch.distributed.init_process_group = init_wrapper(torch.distributed.init_process_group)
torch.distributed.new_group = new_group_wrapper(torch.distributed.new_group)

torch.distributed.isend = send_recv_wrapper(torch.distributed.isend)
torch.distributed.irecv = send_recv_wrapper(torch.distributed.irecv)
torch.distributed.distributed_c10d.isend = torch.distributed.isend
torch.distributed.distributed_c10d.irecv = torch.distributed.irecv
torch.distributed.barrier = lambda *args, **kwargs: None