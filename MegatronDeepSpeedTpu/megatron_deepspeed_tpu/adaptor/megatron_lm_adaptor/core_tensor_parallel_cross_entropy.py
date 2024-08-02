from functools import wraps
import torch
import megatron
from megatron import core

def vocab_parallel_cross_entropy_wrapper(func):
    @wraps(func)
    def wrapper(vocab_parallel_logits, target, label_smoothing=0.0):
        if target.dtype == torch.int64:
            target = target.to(torch.int32)
        return func(vocab_parallel_logits, target, label_smoothing=label_smoothing)
    return wrapper

megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy = \
    vocab_parallel_cross_entropy_wrapper(megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy)
megatron.core.models.common.language_module.language_module.tensor_parallel.vocab_parallel_cross_entropy = \
    vocab_parallel_cross_entropy_wrapper(megatron.core.models.common.language_module.language_module.tensor_parallel.vocab_parallel_cross_entropy)