from typing import Optional
import warnings
import logging

import torch

import torch_tpu
import os

logger = logging.getLogger(__name__)

TORCHTPU_STORAGE_CAST = os.environ.get('TORCHTPU_STORAGE_CAST', None)

def tpu(self, device=None):
    r"""Moves all model parameters and buffers to the tpu.

    This also makes associated parameters and buffers different objects. So
    it should be called before constructing optimizer if the module will
    live on tpu while being optimized.

    Arguments:
        device (int, optional): if specified, all parameters will be
            copied to that device

    Returns:
        Module: self
    """
    device = torch.device('tpu')

    new_obj = self._apply(lambda t: t.tpu(device))
    if torch_tpu.tpu.is_available():
        with torch.no_grad():
            new_obj.cast_weight(device)   
    return new_obj

def to(self, *args, **kwargs):
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    if dtype is not None:
        if dtype.is_complex or dtype == torch.int64:
            raise TypeError('nn.Module.to only accepts floating point or int8 '
                            'dtypes, but got desired dtype={}'.format(dtype))
    def convert(t):
        if convert_to_format is not None and t.dim() in (4, 5):
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
    new_obj = self._apply(convert)
    if torch_tpu.tpu.is_available():
        new_obj.cast_weight(device)
    return new_obj

def cast_weight(self, device):
    def _format_cast(module, class_name):
        if TORCHTPU_STORAGE_CAST in ["ON", "Yes", "1", "yes", "on", "On"]:
            if issubclass(class_name, torch.nn.Conv2d):
                if module.training:
                    module.weight.data = torch_tpu.tpu_format_cast(module.weight.data, 2) #TPU_FORMAT_CONV_W_TRAIN
                    print("Conv parameter transfered with TPU_FORMAT_CONV_W_TRAIN to TPU")
                else:
                    module.weight.data = torch_tpu.tpu_format_cast(module.weight.data, 1)  #TPU_FORMAT_CONV_W_INFER
                    print("Conv parameter transfered with TPU_FORMAT_CONV_W_INFER to TPU")


    if (device is not None) and ("tpu" not in str(device)):
        print(f"{device} no support cast_weight")
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
    torch.nn.Module.cast_weight = cast_weight
    ##TODO: parallel, special module