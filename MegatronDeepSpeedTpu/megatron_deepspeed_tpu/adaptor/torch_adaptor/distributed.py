import torch
from torch import distributed as dist

class Noop:
    def wait(self):
        return None
    
def broadcast_wrapper(func):
    def wrapper(tensor, src, group=None, async_op=False):
        if dist.get_world_size(group=group) == 1:
            return Noop()
        if async_op:
            print("async_op=True is not supported. Use async_op=False instead.")
        if tensor.dtype == torch.int64:
            _tensor = tensor.view(torch.int32)
            ret = func(tensor=_tensor, src=src, group=group, async_op=False)
            return ret
        return func(tensor, src, group=group, async_op=False)
    return wrapper

def reduce_wrapper(func):
    def wrapper(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if dist.get_world_size(group=group) == 1:
            return Noop()
        if async_op:
            print("async_op=True is not supported. Use async_op=False instead.")
        if tensor.dtype in [torch.bool, torch.int8, torch.uint8]:
            _tensor = tensor.int()
            ret = func(tensor=_tensor, dst=dst, op=op, group=group, async_op=False)
            tensor.copy_(_tensor.to(tensor.dtype))
            return ret
        return func(tensor, dst, op=op, group=group, async_op=False)
    return wrapper

def all_reduce_wrapper(func):
    def wrapper(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if dist.get_world_size(group=group) == 1:
            return Noop()
        if async_op:
            print("async_op=True is not supported. Use async_op=False instead.")
        if tensor.dtype in [torch.bool, torch.int8, torch.uint8]:
            _tensor = tensor.int()
            ret = func(tensor=_tensor, op=op, group=group, async_op=False)
            tensor.copy_(_tensor.to(tensor.dtype))
            return ret
        return func(tensor, op=op, group=group, async_op=False)
    return wrapper

def all_gather_into_tensor_using_all_gather(output_tensor, input_tensor, group=None, async_op=False):
    print("all_gather_into_tensor is not supported. Using All Gather instead.")
    group_size = dist.get_world_size(group=group)
    if group_size == 1:
        output_tensor.copy_(input_tensor)
        return Noop()
    if async_op:
        print("async_op=True is not supported. Use async_op=False instead.")
    tensor_list = [torch.zeros_like(input_tensor) for _ in range(group_size)]
    handle = dist.all_gather(tensor_list=tensor_list, tensor=input_tensor, group=group, async_op=False)
    output_tensor.copy_(torch.cat(tensor_list, dim=0))
    return handle

torch.distributed.broadcast = broadcast_wrapper(torch.distributed.broadcast)
torch.distributed.reduce = reduce_wrapper(torch.distributed.reduce)
torch.distributed.all_reduce = all_reduce_wrapper(torch.distributed.all_reduce)
torch.distributed.all_gather_into_tensor = all_gather_into_tensor_using_all_gather