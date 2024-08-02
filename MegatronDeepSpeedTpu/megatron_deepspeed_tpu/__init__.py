import torch
from distutils.util import strtobool
from .utils import environ_flag

if not environ_flag('DS_DISABLE_TPU'):
    import torch_tpu
    if environ_flag('ENABLE_TPU_WORKAROUNDS'):
        import tpu_workarounds
    from . import adaptor

from . import debugger
from .utils import set_comm_socket
set_comm_socket()