import os
import torch

here = os.path.dirname(os.path.abspath(__file__))
TPU_PLUGIN_PATH = os.path.join(here, "lib/liblibtorch_plugin.so")
torch.ops.load_library(TPU_PLUGIN_PATH)

from . import tpu

def TPU(dev_id = 0):
  return torch.device(f"privateuseone:{dev_id}")