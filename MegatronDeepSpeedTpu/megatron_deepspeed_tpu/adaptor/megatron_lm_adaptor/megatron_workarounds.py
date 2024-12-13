from functools import wraps
import torch
from torch import Tensor
import megatron
from megatron import core

def _get_global_min_max_time(self, names, reset, barrier, normalizer):
    """Report only min and max times across all ranks."""

    rank_name_to_time = self._get_elapsed_time_all_ranks(names, reset, barrier)
    name_to_min_max_time = {}
    for i, name in enumerate(names):
        rank_to_time = rank_name_to_time[:, i].contiguous()
        # filter out the ones we did not have any timings for
        rank_to_time_gt_0 = rank_to_time > 0.0
        # If the timer exists:
        if rank_to_time_gt_0.float().sum() > 0:
            rank_to_time = rank_to_time[rank_to_time_gt_0]
            name_to_min_max_time[name] = (
                rank_to_time.min().item() / normalizer,
                rank_to_time.max().item() / normalizer,
            )
    return name_to_min_max_time

megatron.core.timers.Timers._get_global_min_max_time = _get_global_min_max_time

def forward_wrapper(func):
    @wraps(func)
    def wrapper(self, input_):
        if input_.dtype == torch.int64:
            input_ = input_.int()
        ret = func(self, input_)
        return ret
    return wrapper

megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = forward_wrapper(megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward)

def compute_language_model_loss_wrapper(func):
    @wraps(func)
    def wrapper(self, labels: Tensor, logits: Tensor) -> Tensor:
        if labels.dtype == torch.int64:
            labels = labels.int()
        ret = func(self, labels, logits)
        return ret
    return wrapper

megatron.core.models.common.language_module.language_module.LanguageModule.compute_language_model_loss = compute_language_model_loss_wrapper(megatron.core.models.common.language_module.language_module.LanguageModule.compute_language_model_loss)

# def vocab_parallel_cross_entropy_wrapper(func):
#     def wrapper(vocab_parallel_logits, target, label_smoothing=0.0):
#         if target.dtype == torch.int64:
#             target = target.to(torch.int32)
#         return func(vocab_parallel_logits, target, label_smoothing=label_smoothing)
#     return wrapper

# megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy = \
#     vocab_parallel_cross_entropy_wrapper(megatron.core.tensor_parallel.cross_entropy.vocab_parallel_cross_entropy)
# megatron.core.models.common.language_module.language_module.tensor_parallel.vocab_parallel_cross_entropy = \
#     vocab_parallel_cross_entropy_wrapper(megatron.core.models.common.language_module.language_module.tensor_parallel.vocab_parallel_cross_entropy)