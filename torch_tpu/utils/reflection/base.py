import torch


class NoneOp:
    pass


class TensorLikeBase:
    def __init__(self, tensor):
        self._tensor = tensor

    def __getitem__(self, item):
        raise NotImplementedError()

    def __setitem__(self, item, value):
        raise NotImplementedError()

    def _wrap_callback(self, func):
        raise NotImplementedError()

    def __getattr__(self, name):
        ret = getattr(self._tensor, name)
        if name == "_tensor":
            return super().__getattr__(name)
        if callable(ret):
            return self._wrap_callback(ret)
        elif isinstance(ret, torch.Tensor):
            return self.__class__(ret)
        return ret

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError()

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

    def __xor__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__xor__, None, [self._tensor, other], {}
        )

    def __rxor__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rxor__, None, [self._tensor, other], {}
        )

    def __ixor__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ixor__, None, [self._tensor, other], {}
        )

    def __lshift__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__lshift__, None, [self._tensor, other], {}
        )

    def __rlshift__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rlshift__, None, [self._tensor, other], {}
        )

    def __ilshift__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ilshift__, None, [self._tensor, other], {}
        )

    def __rshift__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rshift__, None, [self._tensor, other], {}
        )

    def __rrshift__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rrshift__, None, [self._tensor, other], {}
        )

    def __irshift__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__irshift__, None, [self._tensor, other], {}
        )

    def __eq__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__eq__, None, [self._tensor, other], {}
        )

    def __ne__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ne__, None, [self._tensor, other], {}
        )

    def __lt__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__lt__, None, [self._tensor, other], {}
        )

    def __le__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__le__, None, [self._tensor, other], {}
        )

    def __gt__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__gt__, None, [self._tensor, other], {}
        )

    def __ge__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__ge__, None, [self._tensor, other], {}
        )

    def __matmul__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__matmul__, None, [self._tensor, other], {}
        )

    def __rmatmul__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__rmatmul__, None, [self._tensor, other], {}
        )

    def __imatmul__(self, other):
        return self.__torch_function__(
            self._tensor.__class__.__imatmul__, None, [self._tensor, other], {}
        )

    def fake_clone(self):
        return self.__class__(self._tensor)


def get_byte_size(tensor: TensorLikeBase) -> int:
    if isinstance(tensor, TensorLikeBase):
        return tensor.element_size() * tensor.numel()
    else:
        return 0
