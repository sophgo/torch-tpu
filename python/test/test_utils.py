import torch
from typing import Any, List, Union, Callable
from torch import nn
import numpy as np

device = "tpu:0"


class InputIter:
    @staticmethod
    def create(dtype_iter, shape_iter, number_func):
        # t = tqdm()
        dtype_iter = list(dtype_iter)
        for shape in shape_iter:
            for dtype in dtype_iter:
                for data in number_func(shape):
                    data = data.to(dtype=dtype)
                    yield data


class ShapeIter:
    @staticmethod
    def any_shape(dim=2):
        base = [1] * dim
        index = 0
        for i in range(10):
            base = base[:]
            base[index] *= 5
            index = (index + 1) % dim
            yield base


class DTypeIter:
    @staticmethod
    def all():
        return [torch.float32, torch.float16, torch.bfloat16, torch.int8]

    @staticmethod
    def float_type():
        return [torch.float32, torch.float16, torch.bfloat16]

    @staticmethod
    def float_type2():
        return [torch.float32, torch.float16]

    @staticmethod
    def float32():
        return [torch.float32]

    @staticmethod
    def int_type():
        return [torch.int8]


class NumberFunc:
    @staticmethod
    def gauss(domain=[-1, 1]):
        def func(shape):
            num = torch.rand(shape)
            mean = (domain[0] + domain[1]) / 2
            num += mean
            num *= domain[1] - mean
            yield num

        return func

    @staticmethod
    def init_weight():
        from torch.nn.init import xavier_normal_, kaiming_normal_

        def nfunc(shape):
            for func in [xavier_normal_, kaiming_normal_]:
                num = torch.rand(shape)
                func(num)
                yield num

        return nfunc

    @staticmethod
    def linespace(domain=[-1, 1]):
        def func(shape):
            size = np.prod(shape)
            yield from [torch.linspace(domain[0], domain[1], size).reshape(shape)]

        return func

    @staticmethod
    def all(domain=[-1, 1]):
        yield from NumberFunc.gauss(domain=domain)
        yield from NumberFunc.init_weight(domain=domain)


class Evaluator:
    def __init__(self) -> None:
        self.fns: List[
            Callable[[torch.Tensor, torch.Tensor], Union[None, AssertionError]]
        ] = []

    def add_metric(self, fn):
        self.fns.append(fn)
        return self

    def __call__(self, funcs: Union[List[nn.Module], nn.Module], *iters):
        return self.evavlute(funcs, *iters)

    def add_abs_evalute(self, f32_eps=1e-6, f16_eps=0.05, int8_eps=0):
        def abs_evalute(c_data: torch.Tensor, t_data: torch.Tensor, *ipts):
            eps = torch.abs(c_data - t_data)
            max_eps = eps.max().item()
            if t_data.dtype == torch.float32:
                thr = f32_eps
            elif t_data.dtype == torch.float16 or t_data.dtype == torch.bfloat16:
                thr = f16_eps
            elif t_data.dtype == torch.int8:
                thr = int8_eps

            if max_eps <= thr:
                return

            mask = eps > thr

            index = torch.where(mask)[0]

            # failed_input = [i[mask] for i in ipts]
            failed_output = t_data[mask]
            failed_ref = c_data[mask]
            return AssertionError(
                f"max_eps = {max_eps}, {index[:10]}, Failed output: {failed_output[:10]}, Reference: {failed_ref[:10]}"
            )

        return self.add_metric(abs_evalute)

    def evavlute(self, funcs: Union[List[nn.Module], nn.Module], *iters, mem=None):
        if isinstance(funcs, nn.Module):
            funcs = [funcs]

        message = []
        for ipts in zip(*iters):
            for func in funcs:
                try:
                    cpu_data = func(*ipts)
                except Exception as e:
                    print(e)
                    continue

                tpu_data = func(*[i.to(device) for i in ipts])
                if isinstance(cpu_data, torch.Tensor):
                    cpu_data = [cpu_data]
                    tpu_data = [tpu_data]
                for index, (c_data, t_data) in enumerate(zip(cpu_data, tpu_data)):
                    t_data = t_data.to("cpu")

                    for fn in self.fns:
                        ret = fn(c_data, t_data, *ipts)
                        if isinstance(ret, AssertionError):
                            message.append([ret, func, index])
                            if isinstance(mem, list):
                                mem.append({"cpu": c_data, "tpu": t_data})

        if any(message):
            raise AssertionError(
                "\n".join(
                    [f"{func}@{index} Failed: {str(i)}" for i, func, index in message]
                )
            )


def evaluate(funcs: Union[List[nn.Module], nn.Module], *iters):
    if isinstance(funcs, nn.Module):
        funcs = [funcs]

    for ipts in zip(*iters):
        for func in funcs:
            try:
                cpu_data = func(*ipts)
            except Exception as e:
                print(e)
                continue

            tpu_data = func(*[i.to(device) for i in ipts])

            if isinstance(cpu_data, torch.Tensor):
                cpu_data = [cpu_data]
                tpu_data = [tpu_data]
            for c_data, t_data in zip(cpu_data, tpu_data):
                t_data = t_data.to("cpu")
                eps = torch.abs(c_data - t_data)
                # np.testing.assert_allclose(
                # actual, desired, rtol=1e-3, atol=1e-1, verbose=False
                # )

                if eps.max().item() > 1e-3:
                    mask = eps > 1e-3
                    index = torch.where(mask)[0]
                    failed_input = [i[mask] for i in ipts]
                    failed_output = t_data[mask]
                    failed_ref = c_data[mask]
                    shapes = [i.shape for i in ipts]
                    import pdb

                    pdb.set_trace()
                    raise AssertionError(
                        f"{eps.max().item()},{func},{shapes} {index}, {failed_input}, {failed_output}, {failed_ref}"
                    )


def test():
    pass
