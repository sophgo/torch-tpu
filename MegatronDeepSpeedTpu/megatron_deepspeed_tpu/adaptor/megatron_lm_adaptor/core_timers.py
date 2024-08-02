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