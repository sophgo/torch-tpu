from typing import Optional
import warnings
import logging

import torch

import torch_tpu


logger = logging.getLogger(__name__)

def tpu(self, device=None):
    r"""Moves all model parameters and buffers to the npu.

    This also makes associated parameters and buffers different objects. So
    it should be called before constructing optimizer if the module will
    live on npu while being optimized.

    Arguments:
        device (int, optional): if specified, all parameters will be
            copied to that device

    Returns:
        Module: self
    """
    device = torch.device('tpu')
    if torch_tpu.tpu.is_available():
        with torch.no_grad():
            self.cast_weight(device)
    return self._apply(lambda t: t.tpu(device))

def to(self, *args, **kwargs):
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    
    if dtype is not None:
        if dtype.is_complex or dtype == torch.int64:
            raise TypeError('nn.Module.to only accepts floating point or int8 '
                            'dtypes, but got desired dtype={}'.format(dtype))
    if torch_tpu.tpu.is_available():
        with torch.no_grad():
            self.cast_weight(device)

    def convert(t):
        if convert_to_format is not None and t.dim() in (4, 5):
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

    return self._apply(convert)

def cast_weight(self, device):
    #TODO weight costom convert
    def _format_cast(module, class_name):
        if issubclass(class_name, torch.nn.Conv2d):
            #TODO: weight reorder
            pass

    if device is None or "tpu" not in str(device):
        return

    current_class = self.__class__
    _format_cast(self, current_class)

    if not self.children:
        return 

    for sub_module in self.children():
        if isinstance(sub_module, torch.nn.Module):
            sub_module.cast_weight(device)



def apply_module_patch():
    torch.nn.Module.tpu = tpu
    torch.nn.Module.to = to
    torch.nn.Module.cast_weight = cast_weight #TODO
    ##TODO: parallel, special module