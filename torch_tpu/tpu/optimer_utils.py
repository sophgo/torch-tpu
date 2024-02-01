import torch
import torch_tpu

__all__ = ["OpTimer_reset", "is_OpTimer_enabled", "OpTimer_dump",
           "GlobalOpTimer_reset", "GlobalOpTimer_dump"]

def is_OpTimer_enabled(enable):
    #TODO
    pass

def OpTimer_reset():
    return torch_tpu._C._Timer_OpReset()

def OpTimer_dump():
    return torch_tpu._C._Timer_OpDump()

def GlobalOpTimer_reset():
    return torch_tpu._C._Timer_GlobalReset()


def GlobalOpTimer_dump():
    return torch_tpu._C._Timer_GlobalDump()