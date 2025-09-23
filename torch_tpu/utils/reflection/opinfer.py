"""
given function, and args, kwargs, opinfer can:

1. infer tensor-like
2. do time simulation, return timestamps

"""

from .base import NoneOp
from .config import ArchConfig, GlobalConfig, get_dtypesize
from typing import Dict, Callable
import numpy as np

from torch_tpu.utils.reflection.base import TensorLikeBase, get_byte_size

import torch
from . import time_model as tm


class OpInfer:
    def __init__(self, func, args, kwargs, ret=None):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.ret = ret

    def tensor_infer(self) -> TensorLikeBase:
        if not isinstance(self.ret, NoneOp):
            return self.ret
        raise NotImplementedError(
            f"{self.func.__name__} is not implemented, try extend OpInfer"
        )

    def time_infer(self, config: GlobalConfig) -> tm.SpanContainer:
        return tm.Chip(name=f"unknown_{self.func.__name__}")

    @classmethod
    def from_function(cls, operation_func, args, kwargs, ret=None):
        """
        优先从子类中 retrieve Operation 的子类，如果没有则创建新的 Operation 子类
        """
        for sub_class in cls.sub_classes.values():
            if sub_class.is_target_class(operation_func, args, kwargs):
                return sub_class(operation_func, args, kwargs, ret)
        return cls(operation_func, args, kwargs, ret)

    @classmethod
    def is_target_class(cls, operation_func, args, kwargs):

        if cls.target_function == operation_func:
            return True
        if cls.target_funcname == operation_func.__name__:
            return True

        return cls.__name__ == operation_func.__name__

    sub_classes: Dict[str, "OpInfer"] = {}

    target_function: Callable = None
    target_funcname: str = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.sub_classes[cls.__name__] = cls


class GetItemInfer(OpInfer):
    target_function = torch.Tensor.__getitem__
    target_funcname = "__getitem__"
    pass


class SetItemInfer(OpInfer):
    target_function = torch.Tensor.__setitem__
    target_funcname = "__setitem__"
    pass


class RMSNormInfer(OpInfer):
    target_function = torch.ops.my_ops.rmsnorm_forward
    target_funcname = "rmsnorm_forward"

    def tensor_infer(self):
        return torch.empty_like(self.args[0])

    def time_infer(self, config: GlobalConfig):
        arch = config.arch

        element_size = self.args[0].numel()
        byte_size = get_byte_size(self.args[0])

        flops = 4 * element_size

        tiu_cycles = flops / arch.core_num / arch.tiu.eu_num
        # breakpoint()
        # element_size / arch.core_num / arch.npu_num /

        gdma_cycles = byte_size / (arch.dram_bw_us * arch.est.ddr_utilization)

        return tm.Chip().add_child(
            tm.Core().add_child(
                tm.Pipeline()
                .add_child(tm.TiuSpan.from_duration(tiu_cycles))
                .add_child(tm.GdmaSpan.from_duration(gdma_cycles))
            ),
        )


class MatMulGptqInfer(OpInfer):
    target_function = torch.ops.my_ops.matmul_gptq_forward
    target_funcname = "matmul_gptq_forward"

    # def tensor_infer(self):
    #     raise NotImplementedError()

    def time_infer(self, config: GlobalConfig):
        arch = config.arch

        x = self.args[0]
        q_weight = self.args[1]
        bias = self.args[2]
        q_scale_zp = self.args[3]
        q_scale_zp = self.args[4]
        group_size = self.args[5]
        weight_bits = self.args[6]
        output = self.args[7]

        M, K, N = x.shape[0], x.shape[1], q_weight.shape[0]

        flops = ...
        byte_size = (
            get_byte_size(x)
            + get_byte_size(q_weight)
            + get_byte_size(bias)
            + get_byte_size(q_scale_zp)
            + get_byte_size(q_scale_zp)
            + get_byte_size(group_size)
            + get_byte_size(weight_bits)
            + get_byte_size(output)
        )

        tiu_cycles = flops / arch.core_num / arch.tiu.eu_num
        gdma_cycles = byte_size / (arch.dram_bw_us * arch.est.ddr_utilization)

        return tm.Chip().add_child(
            tm.Core().add_child(
                tm.Pipeline()
                .add_child(tm.TiuSpan.from_duration(tiu_cycles))
                .add_child(tm.GdmaSpan.from_duration(gdma_cycles))
            ),
        )


class ToInfer(OpInfer):
    target_function = torch.Tensor.to
    target_funcname = "to"

    def tensor_infer(self):
        assert isinstance(self.args[0], TensorLikeBase)
        return self.args[0].fake_clone()
