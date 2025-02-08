#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import torch
from compile.FxGraphConvertor import fx2mlir
from torch.fx._symbolic_trace import symbolic_trace
import argparse
import numpy as np
import os
import torch.nn as nn
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

class FX_IR_TESTER(object):
    ID = 0
    CURRENT_CASE = ""
    def __init__(self,
                args):

        self.test_cases = {"Convolution":self.test_Conv,
                            "Convbackward":self.test_Conv_backward,
                            "bnbwd":self.test_bn_backward,
                            # "maxpoolwithmask":self.test_maxpoolwithmask,
                            # "batchnorm":self.test_batchnorm,
                            # "maxpoolwithmask_bwd":self.test_maxpoolwithmask_bwd,
                            "maxpoolwithmask_full": self.test_maxpoolwithmask_full,
                            "batchnormbwd":self.test_batchnormbwd,}
        self.args = args

    def convert_module_fx(
        self,
        submodule_name: str,
        module: torch.fx.GraphModule,
        args,
        bwd_graph:bool,
        input_data:dict,
        ref_data:dict
    ):
        c = fx2mlir(submodule_name, self.args, False, [])
        return c.convert(module)

    def generate_random(self, shape, dtype='float32', min=-1, max=1):
        scale = max - min
        return (np.random.rand(*shape) * scale + min).astype(dtype)

    def create_random_input(self, shapes, descs):
        if len(descs) == 0:
            inputs = [self.generate_random(s) for s in shapes]
        else:
            inputs = list()
            for i in range(len(shapes)):
                inputs.append(
                    self.generate_random(shapes[i], descs[i].dtype, descs[i].min, descs[i].max))
        return [torch.from_numpy(inp) for inp in inputs]

    def test_single(self, case: str):
        np.random.seed(0)
        torch.manual_seed(7)
        FX_IR_TESTER.ID = 0
        FX_IR_TESTER.CURRENT_CASE = case
        print("Test: {}".format(case))
        if case in self.test_cases:
            os.makedirs(case, exist_ok=True)
            os.chdir(case)
            func= self.test_cases[case]
            func()
            print("====== TEST {} Success ======".format(case))
        else:
            raise RuntimeError("case [{}] is not exist".format(case))
        
    def test_all(self):
        for case in self.test_cases.keys():
            self.test_single(case)

    def trace_and_test(self,in_shapes, torch_model: nn.Module, descs = [], use_cos: bool = False, input_info = None):
        model_name = "{}_{}".format(self.CURRENT_CASE, FX_IR_TESTER.ID)
        FX_IR_TESTER.ID += 1
        inputs = self.create_random_input(in_shapes, descs)
        fx_module = symbolic_trace(torch_model)
        FakeTensorProp(fx_module).propagate(*inputs)
        res = fx_module(*inputs)
        input_ref = {}
        i=0
        for node in fx_module.graph.nodes:
            if node.op == "placeholder":
                name = node.name
                input_ref[name] = inputs[i].detach().numpy()
                i+=1
        np.savez('input_ref.npz', **input_ref)
        output_ref = {}
        for i in range(len(res)):
            output_ref[str(i)] = res[i].detach().numpy()
        np.savez('output_ref.npz', **output_ref)
        self.convert_module_fx(model_name,fx_module,self.args,False,input_ref,output_ref)

    def test_Conv_backward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z):
                out0,out1,out2 = torch.ops.aten.convolution_backward(x, y, z, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
                out2 = None
                return [out0,out1]

        self.trace_and_test([[8, 256, 56, 56],[8, 64, 56, 56],[256, 64, 1, 1]], Model())

    def test_Conv(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y):
                res = torch.ops.aten.convolution.default(x, y, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
                return res

        self.trace_and_test([[1,3,16,16],[3,3,3,3]], Model())

    def test_bn_backward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z,w,a,b,c):
                out0,out1,out2 = torch.ops.aten.native_batch_norm_backward(x, y, z, w,a,b,c, False, 1e-5, [True, True, True])
                return [out0,out1,out2]

        self.trace_and_test([[8,3,16,16],[8,3,16,16],[3],[3],[3],[3],[3]], Model())

    def test_maxpoolwithmask(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                out0,out1 = torch.ops.aten.max_pool2d_with_indices(x,[3,3],[1,1],[1,1],[1,1],False)
                #max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
                return [out0,out1]

        self.trace_and_test([[8,3,16,16]], Model())

    def test_batchnorm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, inp,mean,bias,rm,rv):
                out = torch.ops.aten._native_batch_norm_legit_functional(inp,mean,bias,rm,rv,True,0.1, 1e-5)
                out0 = out[0]
                out1 = out[1]
                out2 = out[2]
                out3 = out[3]
                out4 = out[4]
                return [out0,out1,out2,out3,out4]

        self.trace_and_test([[8,3,16,16],[3],[3],[3],[3]], Model())


    def test_maxpoolwithmask_bwd(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z):
                x = x + 1
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(x,y,[3,3],[2,2],[1,1],[1,1],False, z)
                return [out0]

        self.trace_and_test([[8,64,56,56],[8,64,112,112], [8,64,56,56]], Model())


    def test_maxpoolwithmask_full(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1])
                getitem_5 = max_pool2d_with_indices[0]
                getitem_6 = max_pool2d_with_indices[1]
                y        += 1
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(y,x,[3,3],[2,2],[1,1],[1,1],False, getitem_6)
                out0 += 1
                return [out0, getitem_6, getitem_5]

        self.trace_and_test([[8,1,112,112],[8,1,56,56], [8,1,56,56]], Model())

    # def test_maxpoolwithmask_full(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self):
    #             super(Model, self).__init__()

    #         def forward(self, x,y, index):
    #             y += 1
    #             out0 = torch.ops.aten.max_pool2d_with_indices_backward(y,x,[3,3],[2,2],[1,1],[1,1],False, index)
    #             out0 += 1
    #             return [out0]

    #     self.trace_and_test([[8,64,112,112],[8,64,56,56], [8,64,56,56]], Model())


    def test_maxpoolwithmask_full(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y, z):
                x = torch.nn.functional.relu(x)
                y *= 0.5
                max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(x, [3, 3], [2, 2], [1, 1])
                getitem_5 = max_pool2d_with_indices[0]
                getitem_6 = max_pool2d_with_indices[1]
                out0 = torch.ops.aten.max_pool2d_with_indices_backward(y,x,[3,3],[2,2],[1,1],[1,1],False, getitem_6)
                out0 += 1
                return [out0, getitem_6, getitem_5]
        self.trace_and_test([[8,64,112,112],[8,64,56,56], [8,64,56,56]], Model())
        # self.trace_and_test([[8,64,112,112],[8,64,56,56], [8,64,56,56]], Model())
        # self.trace_and_test([[8,64,112,112],[8,64,56,56], [8,64,56,56]], Model())
    # batchnormbwd
    def test_batchnormbwd(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x,y,z,w,a,b,c):
                out0,out1,out2 = torch.ops.aten.native_batch_norm_backward(x, y, z, w,a,b,c, False, 1e-5, [True, True, True])
                return [out0,out1,out2]

        self.trace_and_test([[8,2048,7,7],[8,2048,7,7],[2048],[2048],[2048],[2048],[2048]], Model())
        c = 512
        self.trace_and_test([[8,c,7,7],[8,c,7,7],[c],[c],[c],[c],[c]], Model())
        h = 14
        w = 14
        self.trace_and_test([[8,c,h,w],[8,c,h,w],[c],[c],[c],[c],[c]], Model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1690", choices=['bm1684x', 'bm1690','sg2260'],
                        help="chip name")
    parser.add_argument("--cmp", action='store_true',
                        help="enable cmp")
    parser.add_argument("--fp", default="",help="fp")
    parser.add_argument("--model", default="resnet50",help="model name")
    parser.add_argument("--debug", default="",help="debug")
    parser.add_argument("--batch", default=1,help="batch")
    parser.add_argument("--case", default="",help="test case",choices=['all','Convolution','Convbackward','bnbwd',
                                                                       'maxpoolwithmask','batchnorm','maxpoolwithmask_bwd',
                                                                       'maxpoolwithmask_full','batchnormbwd','opt3_mlp_group'])
    args = parser.parse_args()
    tester = FX_IR_TESTER(args)
    dir = "torch_test_{}".format(args.chip)
    os.makedirs(dir, exist_ok=True)
    os.chdir(dir)
    if args.case == "all":
        tester.test_all()
    else:
        tester.test_single(args.case)
