from functools import wraps
import torch
import megatron
from megatron import core

def forward_wrapper(func):
    @wraps(func)
    def wrapper(self, input_):
        if input_.dtype == torch.int64:
            input_ = input_.int()
        ret = func(self, input_)
        return ret
    return wrapper

megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = forward_wrapper(megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward)