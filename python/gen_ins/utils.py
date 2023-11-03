import ctypes as ct
import os
# from tkinter.messagebox import NO
import torch
top=os.path.dirname(__file__)
torch.ops.load_library(os.path.join(top, "../../build/torch_tpu/libtorch_tpu.so"))
os.environ['FORBID_CMD_EXECUTE']=str(0)
os.environ['FILE_DUMP_CMD'] ="ins"
os.environ['CMODEL_GLOBAL_MEM_SIZE'] = "34359738368"

if os.environ.get('FORBID_CMD_EXECUTE')=='0':
    if 'FORBID_CMD_EXECUTE' in os.environ:
        del os.environ['FORBID_CMD_EXECUTE']
class DumpIns:
    def __init__(self, lib_path = os.path.join(top, "../../third_party/sg2260/libcmodel_firmware.so")) -> None:
        self.libpath = lib_path
        self._lib = ct.cdll.LoadLibrary(lib_path)
    def dump(self, path):
        self._lib.set_file_dump_subdir(path.encode())

class ForwardHack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, inp):
        DumpIns().dump(name)
        return  inp

    @staticmethod
    def backward(ctx, arg1):
        return None, arg1

class BackwardHack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, args):
        ctx.name = name
        return args

    @staticmethod
    def backward(ctx, args):
        print("===========================", ctx.name)
        DumpIns().dump(ctx.name)
        return None, args
