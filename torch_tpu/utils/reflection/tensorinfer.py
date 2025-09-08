from typing import Dict, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_tpu.utils.reflection.torch_inject import TensorLike


import torch


class TensorInfer:
    def infer(self, func, args, kwargs):
        raise NotImplementedError()

    @classmethod
    def from_function(cls, operation_func, args, kwargs):
        """
        优先从子类中 retrieve Operation 的子类，如果没有则创建新的 Operation 子类
        """
        for sub_class in cls.sub_classes.values():
            if sub_class.is_target_class(operation_func, args, kwargs):
                return sub_class(operation_func, args, kwargs)
        return cls(operation_func, args, kwargs)

    @classmethod
    def is_target_class(cls, operation_func, args, kwargs):

        if cls.target_function == operation_func:
            return True
        if cls.target_funcname == operation_func.__name__:
            return True

        return cls.__name__ == operation_func.__name__

    sub_classes: Dict[str, "TensorInfer"] = {}

    target_function: Callable = None
    target_funcname: str = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.sub_classes[cls.__name__] = cls


class GetItemInfer(TensorInfer):
    def infer(self, item):
        raise NotImplementedError()


class RMSNormInfer(TensorInfer):
    def infer(self, tensor: "TensorLike", dim: int, eps: float):
        return torch.empty_like(tensor)
