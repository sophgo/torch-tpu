import ctypes as ct
import os
# from tkinter.messagebox import NO
import torch
torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
os.environ['FORBID_CMD_EXECUTE'] ="1"
os.environ['FILE_DUMP_CMD'] ="ins"

class DumpIns:
    def __init__(self, lib_path = "../../third_party/sg2260/libbmlib.so") -> None:
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
