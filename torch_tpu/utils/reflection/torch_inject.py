from .recorder import enter_subgraph, exit_subgraph
import torch
from torch import Tensor
from torch.nn.modules.module import (
    Module,
    _global_module_registration_hooks,
    _global_buffer_registration_hooks,
)
from torch.nn.parameter import Parameter
from typing import Union
import torch.nn as nn
from functools import wraps

# 导入图相关的类和操作
from .recorder import (
    append_to_graph,
)


def unwrap_tensor(tensor):
    if isinstance(tensor, TensorLike):
        return tensor._tensor
    if isinstance(tensor, tuple):
        return tuple(unwrap_tensor(t) for t in tensor)
    elif isinstance(tensor, list):
        return [unwrap_tensor(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: unwrap_tensor(v) for k, v in tensor.items()}
    return tensor


def wrap_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return TensorLike(tensor)
    elif isinstance(tensor, tuple):
        return tuple(wrap_tensor(t) for t in tensor)
    elif isinstance(tensor, list):
        return [wrap_tensor(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: wrap_tensor(v) for k, v in tensor.items()}
    return tensor


class TensorLike:
    def __init__(self, tensor):
        self._tensor = tensor

    def wrap_callback(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            ret = wrap_tensor(ret)
            append_to_graph(func, [self, *args], kwargs, ret)
            return ret

        return wrapper

    def __getitem__(self, item):
        ret = self._tensor[item]
        ret = TensorLike(ret)
        append_to_graph(
            self._tensor.__class__.__getitem__,
            [],
            {"tensor": self._tensor, "item": item},
            ret,
        )
        return ret

    def __setitem__(self, item, value):
        self._tensor[item] = value
        # 对于 setitem 操作，我们需要特殊处理参数
        append_to_graph(
            self._tensor.__class__.__setitem__,
            [],
            {"tensor": self._tensor, "item": item, "value": value},
            None,
        )

    def __getattr__(self, name):
        ret = getattr(self._tensor, name)
        if name == "_tensor":
            return super().__getattr__(name)
        if callable(ret):
            return self.wrap_callback(ret)
        elif isinstance(ret, torch.Tensor):
            return TensorLike(ret)
        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 第一步：安全地提取 TensorLike 对象中的原始 tensor 数据，避免递归调用
        def extract_data(arg):
            if isinstance(arg, TensorLike):
                # 直接返回原始的 torch.Tensor 对象，避免递归调用
                return arg._tensor
            return arg

        ori_args = args
        ori_kwargs = kwargs
        # 第二步：处理 args 参数
        args = [extract_data(arg) for arg in args]

        # 第三步：处理 kwargs 参数
        if kwargs is not None:
            kwargs = {k: extract_data(v) for k, v in kwargs.items()}
        else:
            kwargs = {}

            # .simulate()

        # 第四步：调用原始函数
        ret = func(*args, **kwargs)
        ret = wrap_tensor(ret)
        append_to_graph(func, ori_args, ori_kwargs, ret)
        return ret

    def __add__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__add__, None, [self._tensor, other], {}
        )

    def __radd__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__radd__, None, [self._tensor, other], {}
        )

    def __iadd__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__iadd__, None, [self._tensor, other], {}
        )

    def __rsub__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rsub__, None, [self._tensor, other], {}
        )

    def __isub__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__isub__, None, [self._tensor, other], {}
        )

    def __sub__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__sub__, None, [self._tensor, other], {}
        )

    def __rsub__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rsub__, None, [self._tensor, other], {}
        )

    def __isub__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__isub__, None, [self._tensor, other], {}
        )

    def __mul__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__mul__, None, [self._tensor, other], {}
        )

    def __rmul__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rmul__, None, [self._tensor, other], {}
        )

    def __imul__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__imul__, None, [self._tensor, other], {}
        )

    def __truediv__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__truediv__, None, [self._tensor, other], {}
        )

    def __rtruediv__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rtruediv__, None, [self._tensor, other], {}
        )

    def __itruediv__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__itruediv__, None, [self._tensor, other], {}
        )

    def __pow__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__pow__, None, [self._tensor, other], {}
        )

    def __rpow__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rpow__, None, [self._tensor, other], {}
        )

    def __ipow__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ipow__, None, [self._tensor, other], {}
        )

    def __mod__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__mod__, None, [self._tensor, other], {}
        )

    def __rmod__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rmod__, None, [self._tensor, other], {}
        )

    def __imod__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__imod__, None, [self._tensor, other], {}
        )

    def __floordiv__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__floordiv__, None, [self._tensor, other], {}
        )

    def __rfloordiv__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rfloordiv__, None, [self._tensor, other], {}
        )

    def __ifloordiv__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ifloordiv__, None, [self._tensor, other], {}
        )

    def __and__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__and__, None, [self._tensor, other], {}
        )

    def __rand__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rand__, None, [self._tensor, other], {}
        )

    def __iand__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__iand__, None, [self._tensor, other], {}
        )

    def __or__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__or__, None, [self._tensor, other], {}
        )

    def __ror__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ror__, None, [self._tensor, other], {}
        )

    def __ior__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ior__, None, [self._tensor, other], {}
        )


class ParameterLike(TensorLike):
    pass


class TorchWrapper:
    @classmethod
    def empty(cls, *args, **kwargs):
        ret = origin_empty(*args, **kwargs)
        return TensorLike(ret)

    @classmethod
    def randn(cls, *args, **kwargs):
        ret = origin_randn(*args, **kwargs)
        return TensorLike(ret)

    @classmethod
    def rand(cls, *args, **kwargs):
        ret = origin_rand(*args, **kwargs)
        return TensorLike(ret)

    @classmethod
    def zeros(cls, *args, **kwargs):
        ret = origin_zeros(*args, **kwargs)
        return TensorLike(ret)

    @classmethod
    def ones(cls, *args, **kwargs):
        ret = origin_ones(*args, **kwargs)
        return TensorLike(ret)

    @classmethod
    def tensor(cls, *args, **kwargs):
        ret = origin_tensor(*args, **kwargs)
        return TensorLike(ret)

    @classmethod
    def all_gather(cls, *args, **kwargs):
        unwrap_args = unwrap_tensor(args)
        unwrap_kwargs = unwrap_tensor(kwargs)
        origin_all_gather(*unwrap_args, **unwrap_kwargs)
        append_to_graph(origin_all_gather, args, kwargs, None)

    @classmethod
    def all_reduce(cls, *args, **kwargs):
        unwrap_args = unwrap_tensor(args)
        unwrap_kwargs = unwrap_tensor(kwargs)
        origin_all_reduce(*unwrap_args, **unwrap_kwargs)
        append_to_graph(origin_all_reduce, args, kwargs, None)


class PostInitHook(type):
    def __call__(cls, *args, **kwargs):
        # 创建实例
        instance = super().__call__(*args, **kwargs)
        # 调用 hook 方法
        if hasattr(instance, "_post_init"):
            instance._post_init(*args, **kwargs)
        return instance


@classmethod
def module_init_subclass(cls, *args, **kwargs):
    # super().__init_subclass__(*args, **kwargs)
    original_init = cls.__init__

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        exit_subgraph()

    cls.__init__ = wrapped_init


class ModuleWrapper(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 注册 forward hooks 来处理子图逻辑
        self._register_forward_hooks()
        enter_subgraph(self.__class__.__name__, "init")

    def _register_forward_hooks(self):
        """注册 forward pre-hook 和 forward hook 来处理子图逻辑"""
        # 注册 forward pre-hook：在 forward 执行前进入子图
        self.register_forward_pre_hook(self._forward_pre_hook)
        # 注册 forward hook：在 forward 执行后退出子图
        self.register_forward_hook(self._forward_hook)

    def _forward_pre_hook(self, module, input):
        """Forward pre-hook：进入子图"""
        # 导入避免循环依赖

        # 获取模块名称
        module_name = module.__class__.__name__
        # 进入子图
        enter_subgraph(module_name, "forward")
        return input

    def _forward_hook(self, module, input, output):
        """Forward hook：退出子图"""
        # 导入避免循环依赖

        # 退出子图
        exit_subgraph()
        return output

    def register_parameter(self, name, param):
        if isinstance(param, TensorLike):
            param_tensor = param._tensor
        else:
            param_tensor = param

        append_to_graph(
            super().register_parameter, [], {"name": name, "param": param}, None
        )
        super().register_parameter(name, param_tensor)

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, (Parameter, ParameterLike)):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                    "(torch.nn.Parameter or None expected)"
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign '{torch.typename(value)}' as child module '{name}' "
                        "(torch.nn.Module or None expected)"
                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get("_buffers")
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                            "(torch.Tensor or None expected)"
                        )
                    for hook in _global_buffer_registration_hooks.values():
                        output = hook(self, name, value)
                        if output is not None:
                            value = output
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)


class ParameterWrapper(nn.Parameter):
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if isinstance(data, TensorLike):
            data = data._tensor

        if type(data) is torch.Tensor or type(data) is origin_parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            ret = torch.Tensor._make_subclass(cls, data, requires_grad)
            ret = ParameterLike(ret)
            return ret

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                f"Creating a Parameter from an instance of type {type(data).__name__} "
                "requires that detach() returns an instance of the same type, but return "
                f"type {type(t).__name__} was found instead. To use the type as a "
                "Parameter, please correct the detach() semantics defined by "
                "its __torch_dispatch__() implementation."
            )
        t._is_param = True
        t = ParameterLike(t)
        return t


origin_parameter = nn.Parameter
origin_parameter_new__ = nn.Parameter.__new__
origin_empty = torch.empty
origin_randn = torch.randn
origin_rand = torch.rand
origin_zeros = torch.zeros
origin_ones = torch.ones
origin_tensor = torch.tensor
origin_all_gather = torch.distributed.all_gather
origin_all_reduce = torch.distributed.all_reduce
# 保存原始的 torch 模块
_original_torch = torch
OriginModule = nn.Module


def inject():
    """注入 torch 包，拦截所有调用"""
    # 创建包装器
    torch.randn = TorchWrapper.randn
    torch.rand = TorchWrapper.rand
    torch.empty = TorchWrapper.empty
    torch.zeros = TorchWrapper.zeros
    torch.ones = TorchWrapper.ones
    torch.tensor = TorchWrapper.tensor
    torch.distributed.all_gather = TorchWrapper.all_gather
    torch.distributed.all_reduce = TorchWrapper.all_reduce
    nn.Module.__init_subclass__ = module_init_subclass
    nn.Module = ModuleWrapper
    nn.Parameter.__new__ = ParameterWrapper.__new__
    nn.Parameter = ParameterWrapper


def restore():
    """恢复原始的 torch 包"""
    torch.randn = origin_randn
    torch.rand = origin_rand
    torch.empty = origin_empty
    torch.zeros = origin_zeros
    torch.ones = origin_ones
    torch.tensor = origin_tensor
    torch.distributed.all_gather = origin_all_gather
    torch.distributed.all_reduce = origin_all_reduce
    nn.Module = OriginModule
    nn.Module.__init_subclass__ = object.__init_subclass__
    nn.Parameter = origin_parameter
    nn.Parameter.__new__ = origin_parameter_new__
