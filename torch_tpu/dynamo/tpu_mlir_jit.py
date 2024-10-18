# -*- coding: utf-8 -*-
import os
import torch
import time
import copy
from argparse import Namespace
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from torch._functorch import compilers
from partition import partition
from TpuMlirModule import TpuMlirModule
from FxGraphConvertor import fx2mlir
from fx_pass import fx_pass_for_bmm_expand
import pdb
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
import numpy as np
from tools.model_transform import get_model_transform
from utils.cache_tool import CacheTool
from tools.gen_rand_input import run_from_mlir
from tools.model_runner import mlir_inference, free_mlir_module, model_inference
from utils.mlir_shell import mlir_opt_for_top, mlir_lowering, mlir_to_model, f32_blobs_compare
from numpy_helper.npz_compare import npz_compare
import gc
import subprocess

args = None
graph_idx = 0

COSINE_THRESHOLD = 0.99
def cosine_similarity(gt_tensors, pred_tensors):
    if not isinstance(gt_tensors, (tuple, list)):
        gt_tensors = (gt_tensors,)
        pred_tensors = (pred_tensors,)
    for gt_tensor, pred_tensor in zip(gt_tensors, pred_tensors):
        gt_tensor = gt_tensor.flatten().to(torch.float32)
        pred_tensor = pred_tensor.flatten().to(torch.float32)
        print(f'ref_outs:{gt_tensor[:8]}')
        print(f'outs:{pred_tensor[:8]}')
        if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
            if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
                return 1.0
        res = torch.nn.functional.cosine_similarity(gt_tensor, pred_tensor, dim=0, eps=1e-6)
        res = res.cpu().detach().item()
        print(f'>>>cos:{res}')
        if res < 0.8:
            print('cmp fail')
    return res

def _get_disc_decomp():
    from torch._decomp import get_decompositions
    aten = torch.ops.aten
    decompositions_dict = get_decompositions(
        [
            # aten.var_mean,
            # aten._adaptive_avg_pool2d_backward,
            # aten.addcmul,
            # aten.avg_pool2d_backward,
            # aten.binary_cross_entropy_with_logits,
            aten.gelu,
            aten.gelu_backward,
            # aten.glu_backward,
            # aten.grid_sampler_2d,
            # aten.hardsigmoid,
            # aten.hardsigmoid_backward,
            # aten.hardswish,
            # aten.hardswish_backward,
            # aten.hardtanh,
            # aten.hardtanh_backward,
            # aten.logsumexp.default,
            # aten.max_pool2d_with_indices_backward,
            # aten.mse_loss,
            # aten.mse_loss_backward,
            # aten.mv,
            # aten.narrow,
            # aten.native_batch_norm,
            # aten.native_batch_norm_backward,
            # aten.native_dropout_backward,
            # aten.native_group_norm,
            aten.native_group_norm_backward,
            # aten.native_layer_norm,
            aten.native_layer_norm_backward,
            # aten.std_mean.correction,
            # aten._softmax,
            aten._softmax_backward_data,
            # aten.stack,
            # aten.t,
            aten.tanh_backward,
            # aten.threshold_backward,
            # aten.transpose.int,
            # aten.tril.default,
            # aten.upsample_bilinear2d.vec,
            # aten.upsample_nearest2d_backward,
            # aten._unsafe_view,
            # aten._native_batch_norm_legit_functional,
            # aten._log_softmax,
            # aten.nll_loss_forward,
            # aten.addmm,
            # aten.leaky_relu,
            # aten.leaky_relu_backward,
            aten.slice_backward,
            # aten.convolution_backward,
            aten.select_backward,
            aten.embedding_dense_backward,
            # aten.select_scatter,
            # aten.slice_scatter,
            aten.sigmoid_backward,
            aten.nll_loss_backward,
            aten._log_softmax_backward_data,
            aten.nll_loss_forward,
            aten.mse_loss,
            aten.mse_loss_backward,
        ]
    )
    return decompositions_dict

def convert_module_fx(
    submodule_name: str,
    module: torch.fx.GraphModule,
    args:Namespace,
    bwd_graph:bool
) -> TpuMlirModule:
    c = fx2mlir(submodule_name, args, bwd_graph)
    return c.convert(module)

def save_fxgraph_dot(name, module):
    if 'enable_dot_graph' in args.debug:
        from torch.fx.passes.graph_drawer import FxGraphDrawer
        g = FxGraphDrawer(module, name)
        with open(f'{name}.svg', "wb") as f:
            f.write(g.get_dot_graph().create_svg())
def unified_compiler(fx_g,example_inputs):
    ## print fx_g
    print('run unified_compiler, original graph:')
    fx_g.graph.print_tabular()
    ## dump fx_g　
    global graph_idx
    graph_str = f'{graph_idx}'
    name = args.model_name + "_"+graph_str
    fx_g.to_folder(name)

    ## get input shape
    input_shape = []
    input_dtype = []

    for node in fx_g.graph.nodes:
        if node.op == 'placeholder':
            shape = list(node.meta['val'].size())
            dtype = node.meta['val'].dtype
            dtype_str = str(dtype).split('.')[-1]
            input_shape.append(shape)
            input_dtype.append(dtype_str)
    ### gen inputs ###
    with maybe_disable_fake_tensor_mode():
        max = 1
        min = -1
        scale = max - min
        input_tensors = []
        for shape in input_shape:
            inputs = (np.random.rand(*shape) * scale + min).astype('float32')
            tensor = torch.from_numpy(inputs)
            input_tensors.append(tensor)

        ## jit trace(module -> pt)
        import importlib
        module = importlib.import_module(f'{name}.module')
        FxModule = getattr(module, 'FxModule')
        torch_model = FxModule()
        pt_name = os.path.join(name,"torch_model.pt")
        torch.jit.trace(torch_model, (input_tensors)).save(pt_name)

        ## pt -> top mlir
        args.input_shapes = input_shape
        # args.mlir =  os.path.join(name,"origin.mlir")
        args.mlir = "test.mlir"
        args.model_def = pt_name
        args.input_types = input_dtype
        cache_tool = CacheTool(args.cache_skip)
        tool = get_model_transform(args)
        tool.model_transform(args.mlir, args.add_postprocess, args.patterns_count)

        ## gen mlir input and ref data
        # run_from_mlir(args)
        subprocess.run(['python', 'gen_rand_input.py', '--mlir', args.mlir])
        # args.test_input = ['input.npz']
        # tool.model_validate(args.test_input, args.tolerance, args.excepts, args.test_result)

        from tools.model_runner import torch_inference
        input_data = dict(np.load("input.npz"))
        ref_data = torch_inference(input_data, args.model_def)
        np.savez("ref_data.npz",**ref_data)
        in_ref_data = 'input.npz'
        if args.cmp:
            tensors = mlir_inference(input_data, args.mlir, True)
            if os.path.exists('ref_data.npz'):
                np.savez('top_ir_out_data.npz', **tensors)
                npz_compare(['top_ir_out_data.npz', 'ref_data.npz', "--tolerance", "0.99,0.98", "-v"])
            else:
                np.savez('ref_data.npz', **tensors)
            del tensors
            free_mlir_module()
            gc.collect()
        tpu_ir = 'tpu_'+args.mlir
        bmodel_path = os.path.join(name, tpu_ir+'.bmodel')
        num_core = 8
        if args.fp == "fp16":
            mlir_lowering(args.mlir, tpu_ir, 'F16', args.chip, num_core = num_core) #F32
        else:
            mlir_lowering(args.mlir, tpu_ir, 'F32', args.chip, num_core = num_core)
        if args.cmp:
            tensors = mlir_inference(input_data, tpu_ir, True)
            ### key adjust in ref_data###
            if args.fp == "fp16":
                new_ref_data = {}
                for key in ref_data.keys():
                    adjust_name = key+"_f32"
                    if adjust_name in tensors:
                        new_ref_data[adjust_name] = ref_data[key]
                    else:
                        new_ref_data[key] = ref_data[key]
                np.savez("ref_data.npz",**new_ref_data)
            np.savez('tpu_ir_out_data.npz', **tensors)
            del tensors
            free_mlir_module()
            gc.collect()
            npz_compare(['tpu_ir_out_data.npz', 'ref_data.npz', "--tolerance", "0.99,0.98", "-v"])

        mlir_to_model(tpu_ir, bmodel_path, 'final_'+args.mlir, opt = 2, debug_cmd = f'--debug_cmd={args.debug}')
        if args.cmp:
            tensors = model_inference(input_data, bmodel_path)
            np.savez('bmodel_out_data.npz', **tensors)
            del tensors
            gc.collect()
            npz_compare(['bmodel_out_data.npz', 'ref_data.npz', "--tolerance", "0.95,0.80", "-v"])
        ## top -> tpu
        import pdb;pdb.set_trace()

    # if args.test_input and cache_tool.do_top_validate(tool.mlir_file, tool.in_f32_npz, args.tolerance, args.debug):
    #     assert (args.test_result)
    #     tool.model_validate(args.test_input, args.tolerance, args.excepts, args.test_result)
    # if not args.debug:
    #     tool.cleanup()

    # tool = TorchTransformer(args.model_name, args.model_def, args.input_shapes,
    #                             args.input_types, args.output_names, preprocessor.to_dict(),
    #                             dynamic=args.dynamic, shape_influencing_input_names=args.shape_influencing_input_names)

def tpu_mlir_compiler(fx_g, example_inputs):
    if 'const_name' not in args.debug:
        time_str = time.strftime("time%Y%m%d%H%M%S", time.localtime())
    else:
        global graph_idx
        time_str = f'{graph_idx}'
        graph_idx += 1
    os.system(f'rm -rf fx_graph_dumped*;mkdir -p {time_str}')
    print('run tpu_mlir_compiler, original graph:')
    #fx_g.graph.print_tabular()
    save_fxgraph_dot(f"fx_g_{time_str}", fx_g)

    for i, node in enumerate(fx_g.graph.nodes):
        print(f'>>> {i}th op, name:', node.name, 'target:',node.target, 'args:', node.args, 'users:', list(node.users.keys()), 'kwargs:', node.kwargs,
              'val:', node.meta['val'] if 'val' in node.meta else 'None')
    if args.only_test_bwd:
        args.only_test_bwd = False
        return make_boxed_func(fx_g.forward)

    fx_g_bk = copy.deepcopy(fx_g)

    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    # FakeTensorProp(fx_g).propagate(*example_inputs)

    if fx_pass_for_bmm_expand(fx_g):
        print('run tpu_mlir_compiler, updated graph:')
        fx_g.graph.print_tabular()

    # gen ref data
    with maybe_disable_fake_tensor_mode():
        inputs = []
        for fake_tensor in example_inputs:
            real_tensor = torch.randn(fake_tensor.shape)
            if fake_tensor.dtype == torch.int64:
                real_tensor = real_tensor.to(torch.int64)
            inputs.append(real_tensor)
        inputs_t = tuple(inputs)
        # res = fx_module(input0,input1)
        name = 'module_fx'
        if os.path.exists(name):
            # rename
            os.mkdir(f'{name}_bwd')
            name = f'{name}_bwd'
        fx_g.to_folder(name)
        if name == 'module_fx':
            from module_fx.module import FxModule
        else:
            from module_fx_bwd.module import FxModule
        mod = FxModule()
        res = mod(*inputs_t)
        # dump res
        output_ref = {}
        for node in fx_g.graph.nodes:
            if node.op == "output":
                output = node.args[0]
                for idx,out in enumerate(output):
                    name = out.name
                    output_ref[name] = res[idx].detach().numpy()
        np.savez('ref_data.npz', **output_ref)
        # dump input
        input_ref = {}
        i = 0
        for node in fx_g.graph.nodes:
            if node.op == "placeholder":
                name = node.name
                input_ref[name] = inputs[i].detach().numpy()
                i+=1
        np.savez('input_ref.npz', **input_ref)

    with compilers._disable_jit_autocast():
        compilers.strip_overloads(fx_g) #删除掉node.target的重载,比如将aten.sum.dim_IntList变为aten.sum
        # for node in fx_g.graph.nodes:
        #     if (
        #         node.target == torch.ops.aten._to_copy
        #         and len(node.args) == 1
        #         and len(node.kwargs) == 1
        #         and "dtype" in node.kwargs
        #     ):
        #         node.target = torch.ops.aten.to
            # if node.target == torch.ops.prims.div:
            #     node.target = torch.ops.aten.div
            # if node.target == torch.ops.aten.alias:
            #     node.target = torch.ops.aten.clone
            # if node.target == torch.ops.prims.var:
            #     node.target = torch.ops.aten.var
            # if node.target == torch.ops.prims.sum:
            #     print('change prims.sum')
            #     node.target = torch.ops.aten.sum
            # if node.target == torch.ops.prims.convert_element_type:
            #     node.target = torch.ops.aten.to
            # if node.target == torch.ops.aten.view:
            #     node.target = torch.ops.aten.reshape

        # for node in fx_g.graph.nodes:
        #     new_kwargs = {}
        #     for k, v in node.kwargs.items():
        #         if isinstance(v, torch.device):
        #             v = v.type # device(type='cuda', index=0)
        #         new_kwargs[k] = v #将device改为字符串形式, why?
        #     node.kwargs = new_kwargs
        # fx_g.graph.lint()
        # fx_g.recompile()

        bwd_graph = len([node for node in fx_g.graph.nodes if node.op == 'placeholder' and node.name == 'tangents_1']) > 0
        # partitioned_module = partition(fx_g, min_block_size = 3)
        # save_fxgraph_dot(f"partitioned_module_{time_str}", partitioned_module)

        # if len(list(partitioned_module.named_children())) > 0:
        #     for name, _ in partitioned_module.named_children():
        #         submodule = getattr(partitioned_module, name)
        #         print(name, 'submodule:', submodule)

        #         tpu_mlir_mod = convert_module_fx(f'{time_str}_{name}', submodule, args, bwd_graph)
        #         if tpu_mlir_mod is not None:
        #             setattr(partitioned_module, name, tpu_mlir_mod)
        # else:
        #     partitioned_module = convert_module_fx(f'{time_str}_main_mod', partitioned_module, args, bwd_graph)
        partitioned_module = convert_module_fx(f'{time_str}_main_mod', fx_g, args, bwd_graph)

    return make_boxed_func(partitioned_module.forward)

#from functorch.compile import min_cut_rematerialization_partition
aot_backend = aot_autograd(bw_compiler = tpu_mlir_compiler,fw_compiler=tpu_mlir_compiler,decompositions=_get_disc_decomp())#fw_compiler=skip_compiler,
# aot_backend = aot_autograd(fw_compiler = unified_compiler)
