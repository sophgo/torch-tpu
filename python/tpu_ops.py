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
import sgdnn_pybind

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
        weight_32IC = torch.zeros((oc * math.ceil(ic / 32) * kh * kw * 32,))
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

        stride_h = get_parameter(stride, 0)
        stride_w = get_parameter(stride, 1)
        dilation_h = get_parameter(dilation, 0)
        dilation_w = get_parameter(dilation, 1)
        padding_h = get_parameter(padding, 0)
        padding_w = get_parameter(padding, 1)

        grad_bias_enable = 0 if bias is None else 1
        sgdnn_pybind.conv_backward(grad_out_np,
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
        grad_weight = torch.from_numpy(grad_weight_np).reshape(weight.shape)
        #grad_weight = torch.from_numpy(grad_weight_np).reshape(weight.shape[1], weight.shape[0], weight.shape[2], weight.shape[3])
        #grad_weight = torch.transpose(grad_weight, 0, 1)
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

class BatchNorm2dFunc(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight, bias, momentum=0.1, eps=1e-5, training=True):
        # reduced_dims = [i for i in range(input.dim()) if i!=1]
        # mean = torch.mean(input, dim=reduced_dims).reshape(weight.shape)
        # var = torch.var(input, dim=reduced_dims, unbiased = False).reshape(weight.shape)
        # invstd = 1/torch.sqrt(var+eps)
        # ctx.save_for_backward(input, weight, mean, invstd)
        # return F.batch_norm(input, running_mean, running_var, weight, bias, True, momentum, eps)
        
        # input
        input_np = np.asarray(input.data.flatten())
        running_mean_np = np.asarray(running_mean.data.flatten())
        running_var_np = np.asarray(running_var.data.flatten())
        weight_np = np.asarray(weight.data.flatten())
        bias_np = np.asarray(bias.data.flatten())
        # output
        batch_mean_np = np.ones(weight.shape, dtype = np.float32)
        batch_invstd_np = np.ones(weight.shape, dtype = np.float32)
        output_np = np.ones(input.shape, dtype = np.float32)
        sgdnn_pybind.batchnorm_forward(input_np,
                                        running_mean_np,
                                        running_var_np,
                                        weight_np,
                                        bias_np,
                                        running_mean_np,
                                        running_var_np,
                                        batch_mean_np,
                                        batch_invstd_np,
                                        output_np,
                                        input.shape[0],
                                        input.shape[1],
                                        input.shape[2],
                                        input.shape[3],
                                        momentum,
                                        eps)
        running_mean = torch.from_numpy(running_mean_np).reshape(weight.shape)
        running_var = torch.from_numpy(running_var_np).reshape(weight.shape)
        batch_mean = torch.from_numpy(batch_mean_np).reshape(weight.shape)
        batch_invstd = torch.from_numpy(batch_invstd_np).reshape(weight.shape)
        output = torch.from_numpy(output_np).reshape(input.shape)
        ctx.save_for_backward(input, weight, batch_mean, batch_invstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, mean, invstd = ctx.saved_tensors
        # test fp16
        # grad_out_np = np.asarray(grad_output.half().data.flatten())
        # input_np = np.asarray(input.half().data.flatten())
        # weight_np = np.asarray(weight.half().data.flatten())
        # mean_np = np.asarray(mean.half().data.flatten())
        # invstd_np = np.asarray(invstd.half().data.flatten())
        # grad_input_np = np.ones(input.shape, dtype = np.float16)
        # grad_weight_np = np.ones(weight.shape, dtype = np.float16)
        # grad_bias_np = np.ones(weight.shape, dtype = np.float16)
        # sgdnn_pybind.batchnorm_backward(grad_out_np,
        #                                   input_np,
        #                                   weight_np,
        #                                   mean_np,
        #                                   invstd_np,
        #                                   grad_input_np,
        #                                   grad_weight_np,
        #                                   grad_bias_np,
        #                                   n,c,h,w,
        #                                   1, 1, 1)
        # test fp32
        grad_out_np = np.asarray(grad_output.data.flatten())
        input_np = np.asarray(input.data.flatten())
        weight_np = np.asarray(weight.data.flatten())
        mean_np = np.asarray(mean.data.flatten())
        invstd_np = np.asarray(invstd.data.flatten())
        grad_input_np = np.ones(input.shape, dtype = np.float32)
        grad_weight_np = np.ones(weight.shape, dtype = np.float32)
        grad_bias_np = np.ones(weight.shape, dtype = np.float32)
        sgdnn_pybind.batchnorm_backward_fp32(grad_out_np,
                                            input_np,
                                            weight_np,
                                            mean_np,
                                            invstd_np,
                                            grad_input_np,
                                            grad_weight_np,
                                            grad_bias_np,
                                            input.shape[0],
                                            input.shape[1],
                                            input.shape[2],
                                            input.shape[3],
                                            1, 1, 1)
        grad_input = torch.from_numpy(grad_input_np).reshape(input.shape)
        grad_weight = torch.from_numpy(grad_weight_np).reshape(weight.shape)
        grad_bias = torch.from_numpy(grad_bias_np).reshape(weight.shape)
        return grad_input, None, None, grad_weight, grad_bias, None, None, None

def BatchNorm2d(input, running_mean, running_var, weight, bias, momentum=0.1, eps=1e-5, training=True):
    return BatchNorm2dFunc.apply(input, running_mean, running_var, weight, bias, momentum, eps, training)

class AvgPool2dFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0,
                     ceil_mode=False, count_include_pad=True, divisor_override=None):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.ceil_mode = ceil_mode
        ctx.count_include_pad = count_include_pad
        ctx.divisor_override = divisor_override
        return F.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        ceil_mode = ctx.ceil_mode
        count_include_pad = ctx.count_include_pad
        divisor_override = ctx.divisor_override

        grad_input = None
        grad_input_np = np.ones(input.shape, dtype = np.float16)

        grad_out_np = np.asarray(grad_output.half().data.flatten())

        kh = get_parameter(kernel_size, 0)
        kw = get_parameter(kernel_size, 1)
        stride_h = kh
        stride_w = kw
        if stride is not None:
            stride_h = get_parameter(stride, 0)
            stride_w = get_parameter(stride, 1)
        pad_h = get_parameter(padding, 0)
        pad_w = get_parameter(padding, 1)

        divisor = kh * kw if divisor_override is None else divisor_override
        sgdnn_pybind.avgpool_backward(grad_out_np,
                                        grad_input_np,
                                        input.shape[0],
                                        input.shape[1],
                                        input.shape[2],
                                        input.shape[3],
                                        grad_output.shape[2],
                                        grad_output.shape[3],
                                        kh, kw,
                                        stride_h, stride_w,
                                        pad_h, pad_w,
                                        ceil_mode,
                                        count_include_pad,
                                        divisor);

        grad_input = torch.from_numpy(grad_input_np).reshape(input.shape)
        return grad_input, None, None, None, None, None, None

def AvgPool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    return AvgPool2dFunc.apply(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

class AdaptiveAvgPool2dFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, output_size):
        ctx.save_for_backward(input)
        ctx.output_size = output_size
        return F.adaptive_avg_pool2d(input, output_size)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        output_size = ctx.output_size

        ih = input.shape[2]
        iw = input.shape[3]
        oh = ih
        ow = iw
        if output_size is not None:
            oh = get_parameter(output_size, 0)
            ow = get_parameter(output_size, 1)

        assert ih % oh == 0
        assert iw % ow == 0

        stride_h = ih//oh
        stride_w = iw//ow
        kh = ih - (oh - 1) * stride_h
        kw = iw - (ow - 1) * stride_w

        grad_input = None
        grad_input_np = np.ones(input.shape, dtype = np.float16)

        grad_out_np = np.asarray(grad_output.half().data.flatten())

        sgdnn_pybind.avgpool_backward(grad_out_np,
                                        grad_input_np,
                                        input.shape[0],
                                        input.shape[1],
                                        ih, iw,
                                        oh, ow,
                                        kh, kw,
                                        stride_h, stride_w,
                                        0, 0,
                                        False,
                                        True,
                                        kh * kw);

        grad_input = torch.from_numpy(grad_input_np).reshape(input.shape)
        return grad_input, None

def AdaptiveAvgPool2d(input, output_size):
    return AdaptiveAvgPool2dFunc.apply(input, output_size);

class MaxPool2dFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.ceil_mode = ceil_mode
        output = F.max_pool2d(input, kernel_size, stride, padding, dilation)
        ctx.save_for_backward(input, output)
        return output

    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        ceil_mode = ctx.ceil_mode

        kh = get_parameter(kernel_size, 0)
        kw = get_parameter(kernel_size, 1)
        stride_h = kh
        stride_w = kw
        if stride is not None:
            stride_h = get_parameter(stride, 0)
            stride_w = get_parameter(stride, 1)
        pad_h = get_parameter(padding, 0)
        pad_w = get_parameter(padding, 1)
        dh = get_parameter(dilation, 0)
        dw = get_parameter(dilation, 1)

        grad_input = None
        grad_input_np = np.ones(input.shape, dtype = np.float32)

        input_np = np.asarray(input.data.flatten())
        output_np = np.asarray(output.data.flatten())
        grad_out_np = np.asarray(grad_output.data.flatten())

        sgdnn_pybind.maxpool_backward_f32(input_np,
                                            output_np,
                                            grad_out_np,
                                            grad_input_np,
                                            input.shape[0],
                                            input.shape[1],
                                            input.shape[2],
                                            input.shape[3],
                                            grad_output.shape[2],
                                            grad_output.shape[3],
                                            kh, kw,
                                            stride_h, stride_w,
                                            pad_h, pad_w,
                                            dh, dw,
                                            ceil_mode);

        grad_input = torch.from_numpy(grad_input_np).reshape(input.shape)
        return grad_input, None, None, None, None, None

def MaxPool2d(input, kernel_size, stride, padding, dilation, ceil_mode):
    return MaxPool2dFunc.apply(input, kernel_size, stride, padding, dilation, ceil_mode)

class EltwiseFunc(Function):
    @staticmethod
    def forward(ctx, input_a, input_b, op_code=1, coeff_a=1, coeff_b=1):
        ctx.op_code = op_code
        ctx.coeff_a = coeff_a
        ctx.coeff_b = coeff_b
        ctx.save_for_backward(input_a, input_b)
        if(op_code==0):
            output = input_a * input_b
        if(op_code==1):
            output = coeff_a * input_a + coeff_b * input_b
        if(op_code==2):
            output = torch.max(input_a, input_b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_a, input_b = ctx.saved_tensors
        grad_input_a = grad_input_b = None
        op_code = ctx.op_code
        coeff_a = ctx.coeff_a
        coeff_b = ctx.coeff_b
        n = grad_output.shape[0]
        c = grad_output.shape[1]
        h = grad_output.shape[2]
        w = grad_output.shape[3]
        input_a_np = np.asarray(input_a.half().data.flatten())
        input_b_np = np.asarray(input_b.half().data.flatten())
        grad_output_np = np.asarray(grad_output.half().data.flatten())
        grad_input_a_np = np.ones(input_a.shape, dtype = np.float16)
        grad_input_b_np = np.ones(input_b.shape, dtype = np.float16)
        sgdnn_pybind.eltwise_backward(input_a_np,
                                        input_b_np,
                                        grad_output_np,
                                        grad_input_a_np,
                                        grad_input_b_np,
                                        n, c, h, w,
                                        op_code,
                                        coeff_a,
                                        coeff_b,
                                        1,1)
        grad_input_a = torch.from_numpy(grad_input_a_np).reshape(input_a.shape)
        grad_input_b = torch.from_numpy(grad_input_b_np).reshape(input_b.shape)
        return grad_input_a, grad_input_b, None, None, None

def Eltwise(input_a, input_b, op_code=1, coeff_a=1, coeff_b=1):
    return EltwiseFunc.apply(input_a, input_b, op_code, coeff_a, coeff_b)

class LinearFunc(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.batch = input.shape[0]
        ctx.in_features = input.shape[1]
        ctx.out_features = weight.shape[0]
        ctx.save_for_backward(input, weight)
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        batch = ctx.batch
        in_features = ctx.in_features
        out_features = ctx.out_features
        input_np = np.asarray(input.half().data.flatten())
        weight_np = np.asarray(weight.half().data.flatten())
        grad_output_np = np.asarray(grad_output.half().data.flatten())
        grad_input_np = np.ones(input.shape, dtype = np.float16)
        grad_weight_np = np.ones(weight.shape, dtype = np.float16)
        grad_bias_np = np.ones(out_features, dtype = np.float16)
        sgdnn_pybind.linear_backward(input_np,
                                       weight_np,
                                       grad_output_np,
                                       grad_input_np,
                                       grad_weight_np,
                                       grad_bias_np,
                                       batch,
                                       in_features,
                                       out_features,
                                       1, 1, 1)
        grad_input = grad_weight = grad_bias = None
        grad_input = torch.from_numpy(grad_input_np).reshape((batch, in_features))
        grad_weight = torch.from_numpy(grad_weight_np).reshape((out_features, in_features))
        grad_bias = torch.from_numpy(grad_bias_np)
        return grad_input, grad_weight, grad_bias

def Linear(input, weight, bias):
    return LinearFunc.apply(input, weight, bias)

class ReluFunc(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        n = input.shape[0]
        c = input.shape[1]
        h = input.shape[2]
        w = input.shape[3]
        input_np = np.asarray(input.half().data.flatten())
        grad_output_np = np.asarray(grad_output.half().data.flatten())
        grad_input_np = np.ones(input.shape, dtype = np.float16)
        sgdnn_pybind.relu_backward(input_np,
                                     grad_output_np,
                                     grad_input_np,
                                     n, c, h, w,
                                     1)   
        grad_input = None
        grad_input = torch.from_numpy(grad_input_np).reshape((n, c, h, w))
        return grad_input

def Relu(input):
    return ReluFunc.apply(input)

class CrossEntropyFunc(Function):
    @staticmethod
    def forward(ctx, input, target, reduction_op=0):
        ctx.reduction_op = reduction_op
        ctx.save_for_backward(input, target)
        return F.cross_entropy(input, target)

    @staticmethod
    def backward(ctx,loss):
        input, target= ctx.saved_tensors
        reduction = ctx.reduction_op
        batch = input.shape[0]
        cls_num = input.shape[1]
        input_np = np.asarray(input.data.flatten())
        target_np = np.asarray(target.data.flatten())
        grad_input_np = np.ones(input.shape,dtype=np.float32)
        sgdnn_pybind.cross_entropy_backward(input_np,
                                           target_np,
                                           grad_input_np,
                                           batch,
                                           cls_num,
                                           reduction)
        grad_input = torch.from_numpy(grad_input_np).reshape((batch, cls_num))
        return grad_input, None, None

def CrossEntropy(input, target, reduction=0):
    return CrossEntropyFunc.apply(input, target, reduction)
