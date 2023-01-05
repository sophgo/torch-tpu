#!/usr/bin/env python3
# -*-coding:utf-8 -*-

import os
import sys
import math

import torch
from torch.nn.modules.utils import _single, _pair, _triple, _quadruple

from tpu_ops import *

backward_ops = [
    "Conv2d",
    "AvgPool2d",
    "MaxPool2d",
]

def assertEqual(got, exp, threshold=0.001, name=None):
    if isinstance(got, list) or isinstance(got, tuple):
        for t, e in zip(got, exp):
            assertEqual(t, e)
    else:
        if (got - exp).abs().max() > threshold:
            # print("compare failed, exp:{} got:{}".format(exp, got))
            print("compare failed, max_difference:{}".format((got - exp).abs().max()))
        else:
            print("{} compare success".format(name if name is not None else ''))

def grad_compare(grad, grad_ref, tolerance, name=None):
    assert(grad.size() == grad_ref.size())
    assertEqual(grad, grad_ref, tolerance, name)

class Test_Backward_Ops(object):

    def __init__(self):
        self.test_function = {
            "Conv2d": self.test_conv2d_backward,
            "BatchNorm2d": self.test_batchnorm2d_backward,
            "AvgPool2d": self.test_avgpool2d_backward,
            "MaxPool2d": self.test_maxpool2d_backward,
        }

    def test_conv2d_backward(self):
        torch.manual_seed(0)
        n = 1
        ic = 1024#1024#2#64
        ih = 14#7#4#64#14
        iw = 14#7#4#64#14
        oc = 2048#1024#1#3#2#64#64
        kh = 1
        kw = 1

        dilation = 1
        groups = 1
        padding = 0
        stride = 2

        input = torch.randn(n, ic, ih, iw, requires_grad = True)
        weight = torch.randn(oc, ic, kh, kw, requires_grad = True)
        bias = torch.randn(oc, requires_grad = True)
        bias_size = [oc, ]

        output = Conv2d(input, weight, bias, stride, padding, dilation, groups)

        grad_output = torch.randn(output.shape)

        output.backward(grad_output)
        grad_bias_enable = False if bias is None else True
        grad_input_ref, grad_weight_ref, grad_bias_ref = torch.ops.aten.convolution_backward(
                                                             grad_output, input, weight, bias_size,
                                                             _pair(stride), _pair(padding), _pair(dilation),
                                                             False, [0], groups, (True, True, grad_bias_enable))

        grad_compare(input.grad, grad_input_ref, 1e-1, "grad_input")
        grad_compare(weight.grad, grad_weight_ref, 1e-1, "grad_weight")
        if bias is not None:
            grad_compare(bias.grad, grad_bias_ref, 1e-1, "grad_bias")

    def test_avgpool2d_backward(self):
        torch.manual_seed(0)
        n = 64
        c = 2048
        h = 7
        w = 7
        kernel_size = 7
        stride = 1
        padding = 0

        input = torch.randn(n, c, h, w, requires_grad = True)
        output = AvgPool2d(input, kernel_size, _pair(stride), _pair(padding), False, True, kernel_size * kernel_size)
        grad_output = torch.randn(output.shape)
        output.backward(grad_output)

        torch_avgpool = nn.AvgPool2d(kernel_size, stride, padding, False, True, kernel_size * kernel_size)
        torch_input = input
        torch_input.grad.zero_()
        torch_output = torch_avgpool(torch_input)
        torch_output.backward(grad_output)

        grad_compare(input.grad, torch_input.grad, 1e-2)

        #from torch.autograd import gradcheck
        #input = torch.randn(n, c, h, w, dtype=torch.double, requires_grad = True)
        #result = gradcheck(AvgPool2d, [input, kernel_size, _pair(stride), _pair(0), False, True, kernel_size * kernel_size])
        #print(result)

    def test_maxpool2d_backward(self):
        torch.manual_seed(0);
        n = 1
        c = 1
        h = 3
        w = 3
        kernel_size = 2
        stride = 1
        padding = 1
        dilation = 1

        input = torch.randn(n, c, h, w, requires_grad = True)
        output = MaxPool2d(input, kernel_size, stride, padding, dilation, False);
        grad_output = torch.randn(output.shape)
        output.backward(grad_output)

        torch_maxpool = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices=True)
        torch_input = input
        torch_input.grad.zero_()
        torch_output, torch_indices = torch_maxpool(torch_input)
        torch_output.backward(grad_output)

'''
        from torch.autograd import gradcheck
        input = torch.randn(n, c, h, w, dtype=torch.double, requires_grad = True)
        print(input)
        result = gradcheck(MaxPool2d, [input, kernel_size, _pair(stride), _pair(padding), _pair(dilation), False])
        print(result)
'''

    def test_batchnorm2d_backward(self):
        torch.manual_seed(0)
        batch_norm_shapes = [
                [100, 64, 2, 2],
                [100, 64, 7, 7],
                [100, 64, 14, 14],
                [100, 64, 28, 28],
                [1, 64, 56, 56],
                [5, 512, 7, 7],
                [5, 64, 112, 112],
                [1, 256, 14, 14],
                [1, 128, 28, 28],
                # [1, 64, 112, 112],
                # [3, 64, 224, 224],
                # [1, 51200, 2, 2],
                # [5, 256, 14, 14],
                # [5, 128, 28, 28],
                # [1, 64, 56, 56],
                # [5, 64, 56, 56],
                # [3, 512, 64, 64],
                # [1, 64, 224, 224],
                ]
        for n, c, h, w in batch_norm_shapes:
            input = torch.rand((n, c, h, w), requires_grad = True)
            weight = torch.rand(c, requires_grad = True)
            bias = torch.rand(c, requires_grad = True)
            mean = torch.mean(input, dim=(0,2,3)).reshape(weight.shape)
            var = torch.var(input, dim=(0,2,3), unbiased = False).reshape(weight.shape)
            invstd = 1/torch.sqrt(var+1e-5)
            output = BatchNorm2d(input, weight, bias)
            grad_output = torch.rand(output.shape)
            output.backward(grad_output)
            grad_input_ref, grad_weight_ref, grad_bias_ref = torch.ops.aten.native_batch_norm_backward(
                grad_output, input, weight, None, None, mean, invstd, True, 1e-5, [True,True,True])
                # grad_output, input, weight, mean, var, None, None, False, 1e-5, [True,False,False])
            grad_compare(input.grad, grad_input_ref, 1)
            grad_compare(weight.grad, grad_weight_ref, 1)
            grad_compare(bias.grad, grad_bias_ref, 1)

if __name__ == "__main__":
    test = Test_Backward_Ops()
    if len(sys.argv) == 2:
        test.test_function[sys.argv[1]]()
    else:
        for op in backward_ops:
            for case in op:
                test.test_function[op]()
                print("====== BACKWARD_OPS {} Test Success ======".format(op))
