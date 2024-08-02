# using host gloo to do communication
from functools import wraps
import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_timeout
from nnmoduletools.module_debugger.utils import print_log

class Noop:
    def wait(self):
        return None

def common_wrapper(func):
    @wraps(func)
    def wrapper(tensor, *args, **kwargs):
        print_log(f"- gloo: {func.__name__}")
        group = kwargs.get("group", None)
        if group is not None and dist.get_world_size(group) == 1:
            return Noop()
        tensor_cpu = tensor.cpu()
        ret = func(tensor_cpu, *args, **kwargs)
        if kwargs.get("async_op", False):
            ret.wait()
            ret = Noop()
        if 'isend' in func.__name__ or 'irecv' in func.__name__:
            ret.wait()
            ret = Noop()
        tensor.data.copy_(tensor_cpu)
        return ret
    return wrapper

def all_gather_into_tensor_using_all_gather(output_tensor, input_tensor, group=None, async_op=False):
    print_log("- all_gather_into_tensor is not supported. Using All Gather instead.")
    group_size = dist.get_world_size(group=group)
    tensor_list = [torch.zeros_like(input_tensor, device="cpu") for _ in range(group_size)]
    handle = dist.all_gather(tensor_list=tensor_list, tensor=input_tensor.cpu(), group=group, async_op=async_op)
    if async_op:
        handle.wait()
        handle = Noop()
    output_tensor.copy_(torch.cat(tensor_list, dim=0))
    return handle

def init_wrapper(func):
    @wraps(func)
    def wrapper(backend=None, init_method=None, timeout=default_pg_timeout, world_size=-1, rank=-1, store=None, group_name="", pg_options=None):
        print_log("Using host gloo to do communication.")
        return func(backend="gloo", init_method=init_method, timeout=timeout, world_size=world_size, rank=rank, store=store, group_name=group_name, pg_options=pg_options)
    return wrapper

def new_group_wrapper(func):
    @wraps(func)
    def wrapper(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None, use_local_synchronization=False):
        print_log("Using host gloo to do communication.")
        return func(ranks=ranks, timeout=timeout, backend="gloo", pg_options=pg_options, use_local_synchronization=use_local_synchronization)
    return wrapper

torch.distributed.broadcast = common_wrapper(torch.distributed.broadcast)
torch.distributed.reduce = common_wrapper(torch.distributed.reduce)
torch.distributed.all_reduce = common_wrapper(torch.distributed.all_reduce)
torch.distributed.isend = common_wrapper(torch.distributed.isend)
torch.distributed.irecv = common_wrapper(torch.distributed.irecv)
torch.distributed.distributed_c10d.isend = torch.distributed.isend
torch.distributed.distributed_c10d.irecv = torch.distributed.irecv
torch.distributed.all_gather_into_tensor = all_gather_into_tensor_using_all_gather
torch.distributed.distributed_c10d.all_gather_into_tensor = all_gather_into_tensor_using_all_gather
torch.distributed.init_process_group = init_wrapper(torch.distributed.init_process_group)
torch.distributed.new_group = new_group_wrapper(torch.distributed.new_group)