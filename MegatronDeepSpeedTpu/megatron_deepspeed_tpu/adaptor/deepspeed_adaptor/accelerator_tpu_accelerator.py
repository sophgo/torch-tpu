import torch
import torch_tpu
import sccl_collectives

import time
import functools

import deepspeed
from deepspeed.accelerator.abstract_accelerator import DeepSpeedAccelerator


class FakeEvent:
    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing
        self.start_time = None
        self.end_time = None

    def record(self):
        if self.enable_timing:
            self.start_time = time.time()

    def synchronize(self):
        if self.enable_timing:
            self.end_time = time.time()

    def elapsed_time(self, end_event):
        if self.enable_timing and self.start_time is not None and end_event.end_time is not None:
            return (end_event.end_time - self.start_time) * 1000  # convert to milliseconds
        else:
            return 0.0

class FakeStream:
    def __init__(self, *args, **kwargs):
        pass

    def synchronize(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
    
    def wait_event(self, event):
        pass

class TPU_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'tpu'
        # self._communication_backend_name = 'sccl' # 1684x
        self._communication_backend_name = 'sccl'

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'tpu'
        return 'tpu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.tpu.device(device_index)

    def set_device(self, device_index):
        if isinstance(device_index, torch.device):
            device_index = device_index.index
        torch_tpu._C._tpu_setDevice(device_index)
        # torch.tpu.set_device(device_index)

    def current_device(self):
        return torch.tpu.current_device()

    def current_device_name(self):
        return 'tpu:{}'.format(torch.tpu.current_device())

    def device_count(self):
        return torch.tpu.device_count()

    def synchronize(self, device_index=None):
        pass

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index == None:
            return torch.tpu.set_rng_state(new_state)
            # return torch.set_rng_state(new_state)
        return torch.tpu.set_rng_state(new_state, device_index)
        # return torch.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index == None:
            return torch.tpu.get_rng_state()
            # return torch.get_rng_state()
        return torch.tpu.get_rng_state(device_index)
        # return torch.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.tpu.manual_seed(seed)
        # return torch.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.tpu.manual_seed_all(seed)
        # return torch.manual_seed(seed)

    def initial_seed(self, seed):
        return torch.tpu.initial_seed(seed)
        # return torch.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.tpu.default_generators[device_index]
        # return torch.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return FakeStream
        # return torch.tpu.Stream
    
    def stream(self, stream):
        return FakeStream()
        # return torch.tpu.stream(stream)

    def current_stream(self, device_index=None):
        return FakeStream()
        # return torch.tpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        # torch.tpu does not support the sync behavior of default stream as cuda
        # use current_stream as workaround
        # see https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
        return FakeStream()
        # return torch.tpu.current_stream(device_index)

    @property
    def Event(self):
        return FakeEvent
        # return torch.tpu.Event

    # Memory management
    def empty_cache(self):
        return
        # return torch.tpu.empty_cache()

    def memory_allocated(self, device_index=None):
        return 0
        # return torch.tpu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return 0
        # return torch.tpu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return 0
        # return torch.tpu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return 0
        # return torch.tpu.memory_reserved(device_index)

    def max_memory_cached(self, device_index=None):
        return 0
        # return torch.tpu.max_memory_reserved(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return 0
        # return torch.tpu.reset_max_memory_reserved(device_index)

    def memory_stats(self, device_index=None):
        mem_stat = {}
        mem_stat['allocated_bytes.all.current'] = 0
        mem_stat['allocated_bytes.all.peak'] = 0
        return mem_stat
        # return torch.tpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        return 0
        # return torch.tpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        return 0
        # return torch.tpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        return 0
        # return torch.tpu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return 4294967296
        # return torch.tpu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return 4294967296
        # return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Misc
    def amp(self):
        return torch.tpu.amp

    def is_available(self):
        return torch.tpu.is_available()

    def range_push(self, msg):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_push(msg)
        return

    def range_pop(self):
        # TODO itt is currently not supported yet
        # return torch.profiler.itt.range_pop()
        return

    def lazy_call(self, callback):
        return callback

    def communication_backend_name(self):
        return self._communication_backend_name

    def is_triton_supported(self):
        return False

    # Graph operations
    def create_graph(self):
        return None

    def capture_to_graph(self, graph, pool=None, stream=None):
        from deepspeed.runtime.utils import noop_context
        return noop_context()

    def replay_graph(self, graph):
        return

    # Data types
    def is_bf16_supported(self):
        return True

    def is_fp16_supported(self):
        return True

    def supported_dtypes(self):
        return [torch.float, torch.half, torch.bfloat16]

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return functools.partial(torch.tensor, dtype=torch.bfloat16, device='tpu')

    @property
    def ByteTensor(self):
        return functools.partial(torch.tensor, dtype=torch.uint8, device='tpu')

    @property
    def DoubleTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='tpu')

    @property
    def FloatTensor(self):
        return functools.partial(torch.tensor, dtype=torch.float, device='tpu')

    @property
    def HalfTensor(self):
        return functools.partial(torch.tensor, dtype=torch.half, device='tpu')

    @property
    def IntTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='tpu')

    @property
    def LongTensor(self):
        return functools.partial(torch.tensor, dtype=torch.int, device='tpu')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('tpu:'):
            return True
        else:
            return False

    def op_builder_dir(self):
        try:
            # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
            # if successful this also means we're doing a local install and not JIT compile path
            from op_builder import __deepspeed__  # noqa: F401 # type: ignore
            return "op_builder"
        except ImportError:
            return "deepspeed.ops.op_builder"

    # dict that holds class name <--> class type mapping i.e.
    # 'AsyncIOBuilder': <class 'op_builder.async_io.AsyncIOBuilder'>
    # this dict will be filled at init stage
    class_dict = {}

    # create an instance of op builder and return, name specified by class_name
    def create_op_builder(self, class_name):
        if class_name in self.class_dict:
            return self.class_dict[class_name]()
        else:
            return None

    # return an op builder class, name specified by class_name
    def get_op_builder(self, class_name):
        if class_name in self.class_dict:
            return self.class_dict[class_name]
        else:
            return None

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def export_envs(self):
        return []
    
def set_tpu_accelerator():
    deepspeed.accelerator.real_accelerator.set_accelerator(TPU_Accelerator())
