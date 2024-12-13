import megatron

def is_kernel_available(*args, **kwargs):
    return False

megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available

def log(self, names, rank=None, normalizer=1.0, reset=True, barrier=False):
    print("Now timers.log() has problems. Disable temporarily.")
megatron.timers.Timers.log = log