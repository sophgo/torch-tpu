from .workaround_helper import *
from functools import partial

torch.Tensor.repeat_interleave = using_cpu_impl(torch.Tensor.repeat_interleave)