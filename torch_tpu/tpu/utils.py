import os
import traceback
import threading

import torch
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

class DeferredNpuCallError(Exception):
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
                raise DeferredNpuCallError(msg) from e

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

def set_device(device):
    if isinstance(device, str) and device.startswith("tpu"):
        device = device.replace('tpu', torch_tpu.tpu.native_device)
    if isinstance(device, (torch_tpu._C.device, torch._C.device)):
        torch_tpu._C._tpu_setDevice(device.index)
    elif isinstance(device, int):
        torch_tpu._C._tpu_setDevice(device)
    elif torch.device(str(device)):
        device_index = torch.device(str(device)).index
        torch_tpu._C._tpu_setDevice(device_index)
    else:
        raise AssertionError("input can not convert to torch.device")

def device_count():
    return torch_tpu._C._tpu_getDeviceCount()

def current_device():
    torch_tpu.tpu._lazy_init()
    return torch_tpu._C._tpu_getDevice()

def is_available():
    if not hasattr(torch_tpu._C, '_tpu_setDevice'):
        return False
    return device_count() > 0