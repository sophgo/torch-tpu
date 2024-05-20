import megatron

def is_kernel_available(*args, **kwargs):
    return False

megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available