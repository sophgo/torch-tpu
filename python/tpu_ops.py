#!/usr/bin/env python3
# -*-coding:utf-8 -*-

'''
https://pytorch.org/docs/master/notes/extending.html
https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py
'''

import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple, _quadruple
from typing import Optional, List, Tuple, Union

sys.path.append(r"./build/sgdnn")
import sgdnn

# support single number or tuple for parameter
def get_parameter(para, idx):
    if isinstance(para,int):
        return para
    elif isinstance(para,tuple):
        return para[idx]

class Conv2dFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1,
                     padding=0, dilation=1, groups=1):
        # ctx is a context object
        # can be used to stash information for backward computation
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient
        # at the top of backward unpack saved_tensors
        # and initialize all gradients w.r.t. inputs to None.
        # Thanks to the fact that additional trailing Nones are ignored,
        # the return statement is simple even when the function has optional inputs.
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        grad_out_np = np.asarray(grad_output.half().data.flatten())
        input_np = np.asarray(input.half().data.flatten())
        weight_np = np.asarray(weight.half().data.flatten())

        # for forward have not import tpu calculate
        # just do reorder in here temporarily
        # can be deleted after when get weight from bmodel
        # [oc, ic, kh, kw] => [1, oc, DIV_UP(ic, 32) * kh * kw, 32]
        oc = weight.shape[0]
        ic = weight.shape[1]
        kh = weight.shape[2]
        kw = weight.shape[3]
        weight_32IC = torch.zeros((oc * math.ceil(ic / 32) * kh * kw * 32,), dtype = torch.float16)
        for oc_idx in range(oc):
            for ic_idx in range(math.ceil(ic / 32)):
                for k_idx in range(kh * kw):
                    for inner in range(32):
                        if ic_idx * 32 + inner >= ic :
                            break
                        src_idx = oc_idx * ic * kh * kw + (ic_idx * 32 + inner) * kh * kw + k_idx;
                        dst_idx = oc_idx * math.ceil(ic / 32) * kh * kw * 32 + ic_idx * kh * kw * 32 + k_idx * 32 + inner;
                        weight_32IC[dst_idx] = weight.data.flatten()[src_idx]
        weight_32IC_np = np.asarray(weight_32IC.half().data)

        grad_input_np = np.ones(input.shape, dtype = np.float16)
        grad_weight_np = np.ones(weight.shape, dtype = np.float16)
        grad_bias_np = np.ones((oc,), dtype = np.float16)

        stride_h = get_parameter(stride, 0);
        stride_w = get_parameter(stride, 1);
        dilation_h = get_parameter(dilation, 0);
        dilation_w = get_parameter(dilation, 1);
        padding_h = get_parameter(padding, 0);
        padding_w = get_parameter(padding, 1);

        grad_bias_enable = 0 if bias is None else 1
        sgdnn.conv_backward(grad_out_np,
                            input_np,
                            weight_32IC_np,
                            grad_input_np,
                            grad_weight_np,
                            grad_bias_np,
                            input.shape[0], input.shape[1],
                            input.shape[2], input.shape[3],
                            grad_output.shape[1],
                            grad_output.shape[2],
                            grad_output.shape[3],
                            groups,
                            weight.shape[2], weight.shape[3],
                            stride_h, stride_w, dilation_h, dilation_w,
                            padding_h, padding_h, padding_w, padding_w,
                            0, 1, 1, grad_bias_enable);

        grad_input = torch.from_numpy(grad_input_np).reshape(input.shape)
        grad_weight = torch.from_numpy(grad_weight_np).reshape(weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3])
        grad_weight = torch.transpose(grad_weight, 0, 1)
        if bias is not None:
            grad_bias = torch.from_numpy(grad_bias_np)

        return grad_input, grad_weight, grad_bias, None, None, None, None

def Conv2d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1):
    return Conv2dFunc.apply(input, weight, bias, stride, padding, dilation, groups)

class tpu_conv2d(nn.Conv2d):
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return Conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        return Conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
