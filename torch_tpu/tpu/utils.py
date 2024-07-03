import os
import traceback
import threading
import warnings
import contextlib
from functools import lru_cache

import torch
from typing import List, Any, Optional, Union
from torch.types import Device
from torch._utils import _get_device_index

import torch_tpu
import torch_tpu._C

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False

def is_initialized():
    r"""Returns whether PyTorch's TPU state has been initialized."""
    return _initialized and not _in_bad_fork

class DeferredTpuCallError(Exception):
    pass

def _lazy_call(callable):
    if _initialized:
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))

def init():
    r"""Initialize PyTorch's TPU state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for TPU functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's TPU methods
    automatically initialize TPU state on-demand.

    Does nothing if the TPU state is already initialized.
    """
    torch_tpu.tpu._lazy_init()

def _lazy_init():
    def _queue_call(queued_calls):
        for queued_call, orig_traceback in queued_calls:
            try:
                queued_call()
            except Exception as e:
                msg = (f"TPU call failed lazily at initialization with error: {str(e)}\n\n"
                        f"TPU call was originally invoked at:\n\n{orig_traceback}")
                raise DeferredTpuCallError(msg) from e

    global _initialized, _original_pid, _queued_calls
    if _initialized or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if _initialized:
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _in_bad_fork:
            from sys import version_info
            if version_info < (3, 4):
                msg = ("To use TPU with multiprocessing, you must use Python "
                       "3.4+ and the 'spawn' start method")
            else:
                msg = ("To use TPU with multiprocessing, you must use the "
                       "'spawn' start method")
            raise RuntimeError(
                "Cannot re-initialize TPU in forked subprocess. " + msg)

        torch_tpu._C._tpu_init()

        _original_pid = os.getpid()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True
        try:
            _queue_call(_queued_calls)
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True

class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch_tpu._C._tpu_getDevice()
        if self.prev_idx != self.idx:
            torch_tpu._C._tpu_setDevice(self.idx)
        torch_tpu.tpu._lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch_tpu._C._tpu_setDevice(self.prev_idx)
        return False    

class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_tpu else -1
        super(device_of, self).__init__(idx)

def set_device(device):
    device_id = _get_device_index(device, optional=True)
    if device_id >= 0:
        torch_tpu._C._tpu_setDevice(device_id)

import json
import subprocess

def get_ip():
    ip = subprocess.run(
        "ifconfig -a|grep inet|grep -w -v 127.0.0.1 |grep -w -v 172.17.0.1 | grep -v inet6|awk '{print $2}'|tr -d 'addr:'",
        shell=True, check=True, stdout=subprocess.PIPE).stdout.decode("utf-8").splitlines()[0]
    assert ip, "unable to get IP"
    return ip

def check_server_count():
    path = os.environ.get("RANK_TABLE_FILE")
    with open(path,'r') as f:
        data = json.load(f)

    server_count = data["server_count"]
    server_list_size = len(data["server_list"])
    assert server_count == server_list_size

def rank_table_valid():
    path = os.environ.get("RANK_TABLE_FILE")
    if path is None:
        print("RANK_TABLE_FILE not set\n")
        return 0
    if not os.path.exists(path):
        print("File not exist\n")
        return 0
    check_server_count()
    return 1

def read_rank_table():
    if rank_table_valid() == 0:
        return None
    path = os.environ.get("RANK_TABLE_FILE")
    with open(path,'r') as f:
        data = json.load(f)

    chip_map = []
    for server_info in data["server_list"]:
        ip = get_ip()
        assert server_info["server_ip"] == ip, "The machine ip is inconsistent with the config ip"
        for device_info in server_info["device"]:
            chip_map.append(int(device_info["device_id"]))
    print(f'[chip_map] {chip_map}')
    return chip_map

@lru_cache(maxsize=1)
def device_count():
    return torch_tpu._C._tpu_getDeviceCount()

def current_device():
    torch_tpu.tpu._lazy_init()
    return torch_tpu._C._tpu_getDevice()

def is_available():
    if not hasattr(torch_tpu._C, '_tpu_setDevice'):
        return False
    return device_count() > 0

def synchronize(device = None):
    r"""Waits for all kernels in all streams on a TPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch_tpu.tpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    torch_tpu.tpu._lazy_init()
    with torch_tpu.tpu.device(device):
        return torch_tpu._C._tpu_synchronize()
    
def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_npu.npu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_tpu.tpu._lazy_init()
    streamdata = torch_tpu._C._tpu_getCurrentStream(
        _get_device_index(device, optional=True))
    return torch_tpu.tpu.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])

def set_stream(stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.
    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch_tpu._C._tpu_setStream(stream_id=stream.stream_id,
                                device_index=stream.device_index,
                                device_type=stream.device_type)


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_tpu.tpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_tpu.tpu._lazy_init()
    streamdata = torch_tpu._C._tpu_getDefaultStream(
        _get_device_index(device, optional=True))
    return torch_tpu.tpu.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def current_blas_handle():
    r"""Return cublasHandle_t pointer to current cuBLAS handle"""

    warnings.warn("TPU does not use blas handle.")
    return None

def can_device_access_peer(device_id, peer_device_id):
    r"""Checks if peer access between two devices is possible.
    """
    device_id = _get_device_index(device_id, optional=True)
    peer_device_id = _get_device_index(peer_device_id, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid devide id")
    if peer_device_id < 0 or peer_device_id >= device_count():
        raise AssertionError("Invalid peer devide id")
    return torch_tpu._C._tpu_canDeviceAccessPeer(device_id, peer_device_id)

def get_arch_list() -> List[str]:
    r"""Returns list TPU architectures this library was compiled for."""
    warnings.warn("TPU does not support get_arch_list now.")
    return []

def get_device_capability(device=None):
    r"""Query the minor and major data of device. Cann does not 
    have a corresponding concept and is not supported. By default, it returns None
    """
    warnings.warn("torch_tpu.tpu.get_device_capability isn't implemented!")
    return None

def get_device_name(device_name=None):
    device_prop = get_device_properties(device_name)
    return device_prop.name

def get_device_properties(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    torch_tpu.tpu._lazy_init()
    return torch_tpu._C._tpu_getDeviceProperties(device_id)

def get_gencode_flags() -> str:
    r"""Returns NVCC gencode flags this library was compiled with."""
    warnings.warn("torch_tpu.tpu.get_gencode_flags isn't implemented!")
    return ""

def get_sync_debug_mode():
    r"""Returns current value of debug mode for tpu synchronizing operations."""
    return torch_tpu._C._tpu_get_sync_debug_mode()

def ipc_collect():
    r"""Force collects GPU memory after it has been released by CUDA IPC.

    .. note::
        Checks if any sent CUDA tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    """
    warnings.warn("TPU not support ipc now !!")
    return None

def memory_usage(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the percent of time over the past sample period during which global (device)
    memory was being read or written. as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    warnings.warn("TPU not support memory_usage api now.")
    return 0

def set_sync_debug_mode(debug_mode):
    r"""Sets the debug mode for tpu synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error.
    """

    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )

    torch_tpu._C._tpu_set_sync_debug_mode(debug_mode)

class StreamContext:
    r"""Context-manager that selects a given stream.

    All TPU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch_tpu.tpu.Stream"]

    def __init__(self, stream: Optional["torch_tpu.tpu.Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch_tpu.tpu.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch_tpu.tpu.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or TPU device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch_tpu.tpu.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch_tpu.tpu.current_stream(cur_stream.device)
        torch_tpu.tpu.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no TPU device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch_tpu.tpu.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch_tpu.tpu.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


@contextlib.contextmanager
def stream(stream):
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    ..Note:: In eager mode stream is of type Stream class while in JIT it is
    an object of the custom class ``torch.classes.tpu.Stream``.
    """
    return StreamContext(stream)

def utilization(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    warnings.warn("TPU not support utilization api now.")
    return 0

def temperature(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the average temperature of the GPU sensor in Degrees C (Centigrades)
        over the past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    warnings.warn("TPU not support temperature api now.")
    # 0 refers to the temperature sensor for the GPU die.
    return 0

def power_draw(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the average power draw of the GPU sensor in mW (MilliWatts)
        over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    warnings.warn("TPU not support power_draw api now.")
    return 0

def clock_rate(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the clock speed of the GPU SM in Hz Hertz over the past sample period as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    warnings.warn("TPU not support clock_rate api now.")
    return 0
