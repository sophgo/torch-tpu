from . import torch_adaptor
from . import deepspeed_adaptor
deepspeed_adaptor.accelerator_tpu_accelerator.set_tpu_accelerator()
from . import megatron_adaptor