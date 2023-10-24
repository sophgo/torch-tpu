import os
import ctypes as ct

here = os.path.dirname(os.path.abspath(__file__))
TPU_PLUGIN_PATH = os.path.join(here, "../../lib/liblibtorch_plugin.so")
class OPTimer:
    def __init__(self, libpath = TPU_PLUGIN_PATH):
        self.lib_path = libpath
        self._lib = ct.cdll.LoadLibrary(libpath)

    def reset(self):
        self._lib.tpu_op_timer_reset()

    def dump(self):
        self._lib.tpu_op_timer_dump()