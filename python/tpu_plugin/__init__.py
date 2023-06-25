import os
import sys
import torch.nn

# setup envs
here = os.path.dirname(os.path.abspath(__file__))
tpu_plugin_libs = os.path.join(here, "lib")
rerun = True
if not 'LD_LIBRARY_PATH' in os.environ:
  os.environ['LD_LIBRARY_PATH'] =":"+tpu_plugin_libs
elif not tpu_plugin_libs in os.environ.get('LD_LIBRARY_PATH'):
  os.environ['LD_LIBRARY_PATH'] +=":"+tpu_plugin_libs
else:
  rerun = False
if rerun:
  os.execv(sys.executable, ['python3'] + [sys.argv[0]])
TPU_PLUGIN_PATH = os.path.join(here, "lib/liblibtorch_plugin.so")
torch.ops.load_library(TPU_PLUGIN_PATH)
 
from . import samples
TPU = "privateuseone"