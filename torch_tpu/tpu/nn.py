import torch
from torch import Tensor
import torch.nn.functional as F

# num_batches_tracked is no-use when momentum is not None 
def BatchNorm2d_forward(self, input: Tensor) -> Tensor:
    self._check_input_dim(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
        # TODO: if statement only here to tell the jit to skip emitting this when it is None
        if self.num_batches_tracked is not None:  # type: ignore[has-type]
            if self.momentum is None:  # use cumulative moving average
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

    r"""
    Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    """
    if self.training:
        bn_training = True
    else:
        bn_training = (self.running_mean is None) and (self.running_var is None)

    r"""
    Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    used for normalization (i.e. in eval mode when buffers are not None).
    """
    return F.batch_norm(
        input,
        # If buffers are not to be tracked, ensure that they won't be updated
        self.running_mean
        if not self.training or self.track_running_stats
        else None,
        self.running_var if not self.training or self.track_running_stats else None,
        self.weight,
        self.bias,
        bn_training,
        exponential_average_factor,
        self.eps,
    )

def apply_nn_patch():
    torch.nn.modules.BatchNorm2d.forward = BatchNorm2d_forward