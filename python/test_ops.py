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
    "BatchNorm2d",
    "AvgPool2d",
    "MaxPool2d",
    "Eltwise",
    "Linear",
    "Relu",
    "CrossEntropy",
]

def assertEqual(got, exp, threshold=0.001, name=None):
    if isinstance(got, list) or isinstance(got, tuple):
        for t, e in zip(got, exp):
            assertEqual(t, e)
    else:
        if (got - exp).abs().max() > threshold:
            #diff_numpy = (got - exp).abs().detach().numpy()
            #exp_numpy = exp.detach().numpy()
            #got_numpy = got.detach().numpy()
            #indices = np.argmax(diff_numpy)
            #print("compare failed, max_difference:{}, exp {}, got {}".format((got - exp).abs().max()), exp_numpy.flatten()[indices], got_numpy.flatten()[indices])
            print("compare failed, max difference:{}".format((got - exp).abs().max()))
            print("\033[1;31;40maverage error\033[0m:{:.4f}".format((got - exp).abs().sum()/exp.numel()))
            print("\033[1;31;40mrelative error\033[0m:{:.4f}%".format(((got-exp)/exp).abs().sum()/exp.numel()*100))
            print("exp {}\ngot {}".format(exp.flatten()[:5], got.flatten()[:5]))
            print("exp {}\n".format(exp.flatten()))
            print("got {}\n".format(got.flatten()))
        else:
            print("{} compare success, max_difference:{}".format(name if name is not None else '',(got - exp).abs().max()))

def grad_compare(grad, grad_ref, tolerance, name=None):
    #assert(grad.size() == grad_ref.size())
    assertEqual(grad, grad_ref, tolerance, name)

class Test_Backward_Ops(object):

    def __init__(self):
        self.test_function = {
            "Conv2d": self.test_conv2d_backward,
            "bn": self.test_batchnorm2d_backward,
            "AvgPool2d": self.test_avgpool2d_backward,
            "MaxPool2d": self.test_maxpool2d_backward,
            "elt": self.test_eltwise_backward,
            "Linear": self.test_linear_backward,
            "Relu": self.test_relu_backward,
            "CrossEntropy": self.test_crossentropy_backward,
        }

    def test_conv2d_backward(self):
        torch.manual_seed(0)
        n = 64
        ic = 3
        ih = 224
        iw = 224
        oc = 64
        kh = 7
        kw = 7

        dilation = 1
        groups = 1
        padding = 3
        stride = 2

        input = torch.randn(n, ic, ih, iw, requires_grad = True)
        weight = torch.randn(oc, ic, kh, kw, requires_grad = True)
        bias = None
        #bias = torch.randn(oc, requires_grad = True)
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
        input_grad = torch.ones(input.grad.shape)
        input_grad.copy_(input.grad.detach())

        torch_avgpool = nn.AvgPool2d(kernel_size, stride, padding, False, True, kernel_size * kernel_size)
        torch_input = input
        torch_input.grad.zero_()
        torch_output = torch_avgpool(torch_input)
        torch_output.backward(grad_output)
        torch_input_grad = torch.ones(torch_input.grad.shape)
        torch_input_grad.copy_(torch_input.grad.detach())

        grad_compare(input_grad, torch_input_grad, 1e-2)

        #from torch.autograd import gradcheck
        #input = torch.randn(n, c, h, w, dtype=torch.double, requires_grad = True)
        #result = gradcheck(AvgPool2d, [input, kernel_size, _pair(stride), _pair(0), False, True, kernel_size * kernel_size])
        #print(result)

    def test_maxpool2d_backward(self):
        torch.manual_seed(0);
        n = 1
        c = 1
        h = 112
        w = 112
        kernel_size = 3
        stride = 2
        padding = 1
        dilation = 1

        input = torch.ones(n, c, h, w, requires_grad = True)

        output = MaxPool2d(input, kernel_size, stride, padding, dilation, False);
        grad_output = torch.randn(output.shape)
        output.backward(grad_output)
        input_grad = torch.ones(input.grad.shape)
        input_grad.copy_(input.grad.detach())

        torch_maxpool = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices=True)
        torch_input = input
        torch_input.grad.zero_()
        torch_output, torch_indices = torch_maxpool(torch_input)
        torch_output.backward(grad_output)
        torch_input_grad = torch.ones(torch_input.grad.shape)
        torch_input_grad.copy_(torch_input.grad.detach())

    def test_batchnorm2d_backward(self):
        torch.manual_seed(0)
        eps = 1e-5
        momentum = 0.1
        resnet50_shapes = [
                [64, 3, 112, 112],
                [64, 64, 56, 56],
                [64, 128, 56, 56],
                [64, 256, 56, 56],
                [64, 128, 28, 28],
                [64, 256, 28, 28],
                [64, 512, 28, 28],
                [64, 256, 14, 14],
                [64, 512, 14, 14],
                [64, 1024, 14, 14],
                [64, 512, 7, 7],
                [64, 2048, 7, 7],
                ]
        for i, [n, c, h, w] in enumerate(resnet50_shapes):
            tpu_input = torch.randn((n, c, h, w), requires_grad = True)
            running_mean = torch.randn(c, requires_grad = False)
            running_var = torch.randn(c, requires_grad = False)
            running_mean.data-=0.5
            weight = torch.rand(c, requires_grad = True)
            bias = torch.rand(c, requires_grad = True)
            tpu_rmean = torch.tensor(running_mean.data)
            tpu_rvar = torch.tensor(running_var.data)
            tpu_input.data-=0.5
            weight.data-=0.5
            bias.data-=0.5
            grad_output = torch.randn((n, c, h, w))
            output = BatchNorm2d(tpu_input, tpu_rmean, tpu_rvar, weight, bias)
            output.backward(grad_output)
            
            # only test backward
            # mean = torch.mean(tpu_input, dim=(0,2,3)).reshape(weight.shape)
            # var = torch.var(tpu_input, dim=(0,2,3), unbiased = False).reshape(weight.shape)
            # invstd = 1/torch.sqrt(var+1e-5)
            # grad_input_ref, grad_weight_ref, grad_bias_ref = torch.ops.aten.native_batch_norm_backward(
            #     grad_output, tpu_input, weight, None, None, mean, invstd, True, 1e-5, [True,True,True])
            # print("case:",i)
            # grad_compare(tpu_input.grad, grad_input_ref, 1e-1)
            # grad_compare(weight.grad, grad_weight_ref, 1e-1)
            # grad_compare(bias.grad, grad_bias_ref, 1e-1)
            # return
        
            # # pytorch reference
            torch_input = torch.rand((n, c, h, w), requires_grad = True)
            torch_rmean = torch.rand(c, requires_grad = False)
            torch_rvar = torch.rand(c, requires_grad = False)
            torch_weight = torch.rand(c, requires_grad = True)
            torch_bias = torch.rand(c, requires_grad = True)
            torch_input.data = tpu_input.data
            torch_rmean.data = running_mean.data
            torch_rvar.data = running_var.data
            torch_weight.data = weight.data
            torch_bias.data = bias.data
            torch_output = torch.nn.functional.batch_norm(
                torch_input,
                torch_rmean,
                torch_rvar,
                torch_weight,
                torch_bias,
                True,
                momentum,
                eps)
            torch_output.backward(grad_output)
            print("case:",i," forward:")
            grad_compare(output, torch_output, 1e-1)
            grad_compare(tpu_rmean, torch_rmean, 1e-1)
            grad_compare(tpu_rvar, torch_rvar, 1e-1)
            print("case:",i," backward:")
            grad_compare(tpu_input.grad, torch_input.grad, 1e-1)
            grad_compare(weight.grad, torch_weight.grad, 1e-1)
            grad_compare(bias.grad, torch_bias.grad, 1e-1)
    
    def test_eltwise_backward(self):
        torch.manual_seed(0)
        op_code = [1,]#[0,1,2]
        resnet50_shapes = [
                [64, 256, 56, 56],
                [64, 512, 28, 28],
                [64, 1024, 14, 14],
                [64, 2048, 7, 7],
                ]
        for i, [n, c, h, w] in enumerate(resnet50_shapes):
            input_a = torch.randn((n, c, h, w), requires_grad = True)
            input_b = torch.randn((n, c, h, w), requires_grad = True)
            torch_input_a = torch.randn((n, c, h, w), requires_grad = True)
            torch_input_b = torch.randn((n, c, h, w), requires_grad = True)
            torch_input_a.data = input_a.data
            torch_input_b.data = input_b.data
            coeff_a = 1 if op_code!=1 else int(torch.randn(1)[0]*10)
            coeff_b = 1 if op_code!=1 else int(torch.randn(1)[0]*10)
            print("case:",i)
            for code in op_code:
                output = Eltwise(input_a, input_b, code, coeff_a, coeff_b)
                grad_output = torch.rand((n, c, h, w))
                if(code==0):
                    print("eltwise op:product")
                    torch_output = torch_input_a * torch_input_b
                if(code==1):
                    print("eltwise op:sum")
                    torch_output = coeff_a * torch_input_a + coeff_b * torch_input_b
                if(code==2):
                    print("eltwise op:max")
                    torch_output = torch.max(torch_input_a, torch_input_b)
                output.backward(grad_output)
                torch_output.backward(grad_output)
                grad_compare(input_a.grad, torch_input_a.grad, 1e-1)
                grad_compare(input_b.grad, torch_input_b.grad, 1e-1)
      
    def test_linear_backward(self):
        torch.manual_seed(0)
        features = [
                [64, 1024, 1000],
                [64, 2048, 1000],
                [64, 4096, 500],
                ]
        for i, [batch, in_features, out_features] in enumerate(features):
            tpu_input = torch.rand((batch, in_features), requires_grad = True)
            weight = torch.rand((out_features, in_features), requires_grad = True)
            bias = torch.rand(out_features, requires_grad = True)
            output = Linear(tpu_input, weight, bias)
            grad_output = torch.rand((batch, out_features))
            output.backward(grad_output)
            # pytorch reference
            torch_linear = nn.Linear(in_features, out_features, True)
            torch_input = torch.rand((batch, in_features), requires_grad = True)
            torch_input.data = tpu_input.data
            torch_linear.weight.data = weight
            torch_linear.bias.data = bias
            torch_output = torch_linear(torch_input)
            torch_output.backward(grad_output)
            print("case:",i)
            grad_compare(tpu_input.grad, torch_input.grad, 1e-1)
            grad_compare(weight.grad, torch_linear.weight.grad, 1e-1)
            grad_compare(bias.grad, torch_linear.bias.grad, 1e-1)
            
    def test_relu_backward(self):
        torch.manual_seed(0)
        resnet50_shapes = [
                [64, 256, 56, 56],
                [64, 512, 28, 28],
                [64, 1024, 14, 14],
                [64, 2048, 7, 7],
                ]
        for i, [n, c, h, w] in enumerate(resnet50_shapes):
            tpu_input = torch.rand((n, c, h, w), requires_grad = True)
            output = Relu(tpu_input)
            grad_output = torch.rand((n, c, h, w))
            output.backward(grad_output)
            #pytorch reference
            torch_input = torch.rand((n, c, h, w), requires_grad = True)
            torch_input.data = tpu_input.data
            torch_output = torch.nn.functional.relu(torch_input)
            torch_output.backward(grad_output)
            print("case:",i)
            grad_compare(tpu_input.grad, torch_input.grad, 1e-1)

    def test_crossentropy_backward(self):
        torch.manual_seed(0)
        reduction = 1 #0
        fc_shapes = [
                [5,  32],
                [64,  500],
                [64, 1000],
                [64, 2000],
                ]
        for i, [batch, cls_num] in enumerate(fc_shapes):
            """
            if label is class indices, need cast to one-hot code or probabilities target
            label = torch.randint(cls_num, (batch,))
            """
            tpu_input = torch.randn((batch, cls_num), requires_grad = True)
            tpu_input.data-=0.5
            target = torch.randn(batch, cls_num)
            target.data-=0.5
            target = target.softmax(dim=1)
            output = CrossEntropy(tpu_input, target, reduction)
            loss = output.data
            output.backward(loss)

            # pytorch reference         
            torch_input = torch.rand((batch, cls_num), requires_grad = True)
            torch_input.data = tpu_input.data
            if reduction:
                torch_loss = torch.nn.functional.cross_entropy(torch_input, target, reduction="sum")
            else:
                torch_loss = torch.nn.functional.cross_entropy(torch_input, target, reduction="mean")
            torch_loss.backward()
            print("case:",i)
            grad_compare(loss, torch_loss, 1e-1)
            grad_compare(tpu_input.grad, torch_input.grad, 1e-1)


if __name__ == "__main__":
    test = Test_Backward_Ops()
    if len(sys.argv) == 2:
        test.test_function[sys.argv[1]]()
    else:
        for op in backward_ops:
            for case in op:
                test.test_function[op]()
                print("====== BACKWARD_OPS {} Test Success ======".format(op))
