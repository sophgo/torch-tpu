import re
import os
import torch
from torch import device as origin_device
from torch.fx.graph import _register_custom_builtin
import torch_tpu

class MetaDevice(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, origin_device)

    def __eq__(self, other):
        return other == origin_device

    def __ne__(self, other):
        return other != origin_device

    def __hash__(self):
        return hash(origin_device)

    def __format__(self, format_spec):
        return f"{origin_device}"

class TPUDevice(metaclass=MetaDevice):
    def __new__(cls, *args, **kwargs):
        chip_map = os.getenv('CHIP_MAP')

        if not chip_map:
            if args and isinstance(args[0], int):
                args = ("tpu", args[0])
            elif isinstance(kwargs.get('device'), int):
                kwargs["device"] = f"tpu:{kwargs.get('device')}"
            return origin_device(*args, **kwargs)

        def _map_local_rank_to_device_idx(index):
            device_mapping = list(map(int, chip_map.split(',')))
            try:
                rank_index = int(index)
            except (TypeError, ValueError):
                return index

            if rank_index < 0:
                raise ValueError(f"Invalid local_rank_idx: {rank_index}")
            return device_mapping[rank_index]

        def process_string_device(device_str):
            match = re.match(r"^tpu:(\d+)$", device_str, re.IGNORECASE)
            if match:
                rank_index = int(match.group(1))
                device_index = _map_local_rank_to_device_idx(rank_index)
                return f"tpu:{device_index}"
            return device_str

        new_args = list(args)
        if new_args:
            if len(new_args) == 1:
                if isinstance(new_args[0], str):
                    new_args[0] = process_string_device(new_args[0])
                elif isinstance(new_args[0], (int, str)) and str(new_args[0]).isdigit():
                    device_index = _map_local_rank_to_device_idx(new_args[0])
                    new_args = ["tpu", device_index]

            elif len(new_args) >= 2:
                if isinstance(new_args[1], (int, str)) and str(new_args[1]).isdigit():
                    new_args[1] = _map_local_rank_to_device_idx(new_args[1])

        args = tuple(new_args)

        if 'device' in kwargs:
            device_val = kwargs['device']
            if isinstance(device_val, str):
                kwargs['device'] = process_string_device(device_val)
            elif isinstance(device_val, int):
                kwargs['device'] = f"tpu:{_map_local_rank_to_device_idx(device_val)}"

        return origin_device(*args, **kwargs)

def apply_device_patch():
    torch.device = TPUDevice
    _register_custom_builtin('device', 'from torch import device', TPUDevice)