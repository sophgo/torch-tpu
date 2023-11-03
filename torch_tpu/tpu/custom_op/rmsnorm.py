import torch
import torch.nn as nn

class RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias, axis, eps):
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)

        torch.ops.my_ops.rmsnorm_forward(x,
                                     scale,
                                     bias,
                                     output,
                                     axis,
                                     eps)
        return output

class RMSNormBlock(nn.Module):
    def __init__(self, hidden_size, axis=-1, eps=1e-8):
        super().__init__()
        scale = torch.empty(hidden_size)
        bias = torch.empty(hidden_size)
        nn.init.normal_(scale, std=0.02)
        nn.init.normal_(bias, std=0.02)
        self.scale = nn.Parameter(scale)
        self.bias = nn.Parameter(bias)
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        return RMSNormFunc.apply(x, self.scale, self.bias, self.axis, self.eps)