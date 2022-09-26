#!/usr/bin/env python3
# -*-coding:utf-8 -*-

'''
https://pytorch.org/docs/master/notes/extending.html
https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py
'''

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import gradcheck

class Conv2dFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1,
                padding=0, dilation=1, groups=1):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = F.grad.conv2d_input(
                    input.shape, weight, grad_output,
                    stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = F.grad.conv2d_weight(
                    input, weight.shape, grad_output,
                    stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3))

        return grad_input, grad_weight, grad_bias
conv2d = Conv2dFunc.apply
