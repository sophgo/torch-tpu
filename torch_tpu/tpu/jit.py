from __future__ import annotations, division
import functools
import inspect
import os
import textwrap
import torch
from typing import (Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast,
                    overload)
T = TypeVar('T')

class KernelInterface(Generic[T]):
    run: T
    def __getitem__(self, grid) -> T:
        assert callable(self.run), "Run method must be callable"
        return cast(T, functools.partial(cast(Callable, self.run), grid=grid))

class LoadDynLib(KernelInterface):
    def __init__(self, fn) -> None:
        self.fn = fn

    def run(self, *args, **kwargs):
        ret = self.fn.run(*args, **kwargs)
        if not ret is None and ret.only_emit_kernel:
            tensors = []
            tensors_index = []
            fp_scalars = []
            fp_scalars_index = []
            fixed_scalars = []
            fixed_scalars_index = []
            arg_index = 0
            for index, tensor in enumerate(args):
                if isinstance(tensor, torch.Tensor):
                    tensors.append(tensor)
                    tensors_index.append(arg_index)
                    arg_index += 1
                elif index not in ret.constexpr_index:
                    if isinstance(tensor, int) or isinstance(tensor, bool):
                        fixed_scalars.append(tensor)
                        fixed_scalars_index.append(arg_index)
                    elif isinstance(tensor, float):
                        fp_scalars.append(tensor)
                        fp_scalars_index.append(arg_index)
                    else:
                        assert False, "don't support"
                    arg_index += 1
            torch.ops.my_ops.dynlib_execute(ret.so_path,
                                        ret.function_name, tensors, tensors_index,
                                        fp_scalars, fp_scalars_index, fixed_scalars,
                                        fixed_scalars_index)
        return ret

def jit(fn: Optional[T] = None):
    def decorator(fn: T):
        return LoadDynLib(fn)

    if fn is not None:
        return decorator(fn)
    else:
        return decorator

def CallCppDynLib(so_path, func_name, *args, **kwargs):
    tensors = []
    tensors_index = []
    fp_scalars = []
    fp_scalars_index = []
    fixed_scalars = []
    fixed_scalars_index = []
    for index, tensor in enumerate(args):
        if isinstance(tensor, torch.Tensor):
            tensors.append(tensor)
            tensors_index.append(index)
        else: 
            if isinstance(tensor, int) or isinstance(tensor, bool):
                fixed_scalars.append(tensor)
                fixed_scalars_index.append(index)
            elif isinstance(tensor, float):
                fp_scalars.append(tensor)
                fp_scalars_index.append(index)
            else:
                assert False, "don't support"
    torch.ops.my_ops.dynlib_execute(so_path, func_name, tensors, tensors_index,
                                fp_scalars, fp_scalars_index, fixed_scalars,
                                fixed_scalars_index)
    return 0
