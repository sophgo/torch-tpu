from .backend import BACKEND

__all__ = [
    "StreamContext",
    "can_device_access_peer",
    "current_blas_handle",
    "current_device",
    "current_stream",
    "default_stream",
    "device",
    "device_count",
    "device_of",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_sync_debug_mode",
    "init",
    "ipc_collect",
    "is_available",
    "is_initialized",
    "memory_usage",
    "set_device",
    "set_stream",
    "set_sync_debug_mode",
    "stream",
    "synchronize",
    "flush",
    "utilization",
    "temperature",
    "power_draw",
    "clock_rate",

    ## sccl related
    "is_rank_table_valid",
    "read_rank_table",
    "set_chip_map",
    "get_topology",

    ## amp related
    "amp",
    "get_amp_supported_dtype",
    "is_autocast_enabled",
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",

    ## rng related
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",

    ## memory management
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",

    ## Streams and events
    "Stream",
    "ExternalStream",
    "Event",

    ## tpu custom funcs
    "OpTimer_reset",
    "is_OpTimer_enabled",
    "OpTimer_dump",
    "GlobalOpTimer_reset",
    "GlobalOpTimer_dump",
    "format_cast",          # tensor format cast

    # "BoolStorage",
    # "ByteStorage",
    # "ShortStorage",
    # "LongStorage",
    # "IntStorage",
    # "HalfStorage",
    # "CharStorage",
    # "DoubleStorage",
    # "FloatStorage",
    # "BoolTensor",
    # "ByteTensor",
    # "CharTensor",
    # "DoubleTensor",
    # "FloatTensor",
    # "HalfTensor",
    # "IntTensor",
    # "LongTensor",
    # "ShortTensor",
    # "config",
    # "matmul",
    # "conv",
]

from typing import Tuple
import torch
import torch_tpu

from . import amp

from .memory import *  # noqa: F403

from .random import * # noqa: F403


from .utils import ( _lazy_call, _lazy_init, init,is_initialized, is_available,
                    device, device_of, device_count, current_device, set_device, read_rank_table, get_topology,
                    is_rank_table_valid, set_chip_map, synchronize, flush, current_stream, set_stream, default_stream,
                    StreamContext, current_blas_handle, can_device_access_peer, get_arch_list,
                    get_device_capability, get_device_name, get_device_properties,
                    get_gencode_flags, get_sync_debug_mode, ipc_collect,
                    memory_usage, set_sync_debug_mode, utilization, temperature,
                    power_draw, clock_rate
                    )
if BACKEND == "SG2260":
    from .streams import Stream, ExternalStream, Event
    from .bmodel_runtime import BmodelRunner, dtype_map

from .autocast_utils import *  # noqa: F403
from .optimer_utils import * # noqa: F403
from .custom_op import *

default_generators: Tuple[torch._C.Generator] = ()
