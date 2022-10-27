#!/usr/bin/env python3
# -*-coding:utf-8 -*-

import os
import sys
import math

import torch
from torch.nn.modules.utils import _single, _pair, _triple, _quadruple

from tpu_ops import Conv2d

backward_ops = [
    "Conv2d",
]

def assertEqual(got, exp, threshold=0.001):
    if isinstance(got, list) or isinstance(got, tuple):
        for t, e in zip(got, exp):
            assertEqual(t, e)
    else:
        if (got - exp).abs().max() > threshold:
            print("compare failed, exp:{} got:{}".format(exp, got))

def grad_compare(grad, grad_ref, tolerance):
    assert(grad.size() == grad_ref.size())
    assertEqual(grad, grad_ref, tolerance)

class Test_Backward_Ops(object):

    def __init__(self):
        self.test_function = {
            "Conv2d": self.test_backward_conv2d,
        }

    def test_backward_conv2d(self):
        torch.manual_seed(0)
        n = 1#2#64
        ic = 512#2#64
        ih = 28#7#4#64#14
        iw = 28#7#4#64#14
        oc = 256#2#64#64
        kh = 3#1
        kw = 3#1
        input = torch.randn(n, ic, ih, iw, requires_grad = True)
        weight = torch.randn(oc, ic, kh, kw, requires_grad = True)
        #bias = torch.randn(oc, requires_grad = True)
        bias = None
        bias_size = [oc, ]
        dilation = 1
        padding = 1
        stride = 2
        groups = 1
        output = Conv2d(input, weight, bias, stride, padding, dilation, groups)
        grad_output = torch.randn(output.shape)
        output.backward(grad_output)
        grad_bias_enable = False if bias is None else True
        grad_input_ref, grad_weight_ref, grad_bias_ref = torch.ops.aten.convolution_backward(
                                                             grad_output, input, weight, bias_size,
                                                             _pair(stride), _pair(padding), _pair(dilation),
                                                             False, [0], groups, (True, True, grad_bias_enable))

        grad_compare(input.grad, grad_input_ref, 1e-2)
        grad_compare(weight.grad, grad_weight_ref, 1e-2)
        if bias is not None:
            grad_compare(bias.grad, grad_bias_ref, 1e-2)

if __name__ == "__main__":
    test = Test_Backward_Ops()
    if len(sys.argv) == 2:
        test.test_function[sys.argv[1]]()
    else:
        for op in backward_ops:
            for case in op:
                test.test_function[op]()
                print("====== BACKWARD_OPS {} Test Success ======".format(op))
