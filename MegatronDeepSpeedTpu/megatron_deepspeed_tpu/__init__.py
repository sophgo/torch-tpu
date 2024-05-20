import os
import torch

if os.environ.get('DS_DISABLE_TPU', '0') == '0':
    import torch_tpu

    if os.environ.get('TPU_DISABLE_WORKAROUNDS', '0') == '0':
        from . import tpu_workarounds
    
    from . import adaptor

from . import debugger
from .utils import set_comm_socket
set_comm_socket()