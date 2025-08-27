import torch
import torch_tpu

__all__ = ["OpTimer_reset", "is_OpTimer_enabled", "OpTimer_dump",
           "OpTimer_pause", "OpTimer_start"]

def is_OpTimer_enabled(enable):
    #TODO
    pass

def OpTimer_reset():
    return torch_tpu._C._Timer_OpReset()

def OpTimer_dump():
    return torch_tpu._C._Timer_OpDump()

def OpTimer_pause():
    return torch_tpu._C._Timer_OpPause()


def OpTimer_start():
    return torch_tpu._C._Timer_OpStart()