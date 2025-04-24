import torch
import json
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from torch._functorch import compilers
from functorch.compile import min_cut_rematerialization_partition
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from typing import List
try:
    from tpu_mlir.python.tools.train.FxGraphConverter import fx2mlir
    from tpu_mlir.python.tools.train.fx_pass import fx_pass_for_bmm_expand
except ImportError:
    raise RuntimeError("Error: This feature requires tpu_mlir package")

import os
from torch_tpu.tpu.bmrt import BmodelModule

######################## dummy compiler
_DummyCompilerPath = "./DummyCompiler"
class dummyModule(torch.nn.Module):
    
    def __init__(self, fx_g, suffix='fwd'):
        super(dummyModule, self).__init__()
        self.fx_g = fx_g
        self.graph_name = suffix
        self.inp_file = os.path.join(_DummyCompilerPath, f'fx_{self.graph_name}_inputs.npz')
        self.out_file = os.path.join(_DummyCompilerPath, f'fx_{self.graph_name}_outputs.npz')
        self.inp_names = []
        self.out_names = []
        for i, node in enumerate(fx_g.graph.nodes):
            if node.op == 'placeholder':
                self.inp_names.append(node.name)
        output_node = [i for i in self.fx_g.graph.nodes if i.op == 'output' and len(i.args) > 0][0]
        for node in output_node.args[0]:
            if node is None: continue
            if '_to_copy' in node.name or 'clone' in node.name: ## this is should not be here, we should change compiler's code
                pre_node = node.args[0]
                self.out_names.append(pre_node.name)
            else:
                self.out_names.append(node.name)

    def forward(self, *inputs):
        import numpy as np
        inputs_dict = {}
        for i, inp in enumerate(inputs):
            name = self.inp_names[i]
            inputs_dict[name] = inp.numpy() if inp is not None else inp
        np.savez(self.inp_file, **inputs_dict)
        print(f'[dummyModule-forward] save inputs to {self.inp_file}')

        outputs_dict = {}
        outputs = self.fx_g(*inputs)
        idx = 0
        for i, out in enumerate(outputs):
            if out is None: continue
            name = self.out_names[idx]
            outputs_dict[name] = out.numpy() if out is not None else out
            idx += 1
        np.savez(self.out_file, **outputs_dict)
        print(f'[dummyModule-forward] save outputs to {self.out_file}')

        return outputs

def save_fx_graph(fx_g : torch.fx.GraphModule, suffix : str = 'fwd'):
    save_txt_f = os.path.join(_DummyCompilerPath, f"graph_table_{suffix}.txt")
    save_pth_f = os.path.join(_DummyCompilerPath, f"graph_module_{suffix}.pth")
    from contextlib import redirect_stdout
    with open(save_txt_f, "w") as f:
        with redirect_stdout(f):
            fx_g.graph.print_tabular()
    torch.save(fx_g, save_pth_f)

def load_fx_graph(path):
    fx_g_loaded = torch.load(path)
    return fx_g_loaded

def dummy_graph_compiler(fx_g : torch.fx.GraphModule, example_inputs : List[torch.Tensor]):
    r''' just save fx graph in txt file, and return fx forward func
    '''
    os.makedirs(_DummyCompilerPath, exist_ok=True)
    is_bwd_graph = len([node for node in fx_g.graph.nodes 
                        if node.op == 'placeholder' and node.name == 'tangents_1']) > 0
    suffix = 'bwd' if is_bwd_graph else 'fwd'
    FakeTensorProp(fx_g).propagate(*example_inputs)
    #fx_pass_for_bmm_expand(fx_g)
    with compilers._disable_jit_autocast():
        compilers.strip_overloads(fx_g) #Remove overloading of node.target, such as aten.sum.dim_intlist to aten.sum
        save_fx_graph(fx_g, suffix)
        return dummyModule(fx_g, suffix).forward

######################## tpu compiler
def _clean_intermidate_file():
    print("clean compiler intermidate files ...")
    os.system("rm compiler*")
    os.system("rm *.mlir")
    os.system("rm *.npz")
    os.system("rm *.json")
    os.system("rm group_before.txt")

def tpu_mlir_fwd_compiler(fx_g : torch.fx.GraphModule, example_inputs : List[torch.Tensor] ):
    save_path = f"tmp_fwd"
    use_f16 = True
    dtype = 'f16' if use_f16 else 'f32'
    bmodel_path = f"tmp/{save_path}_{os.environ['CHIP']}_{dtype}_tpu.bmodel"

    compiled = os.path.isfile(bmodel_path)
    if not compiled:
        tpu_compiler = fx2mlir(submodule_name = save_path, chip = os.environ['CHIP'], bwd_graph= False, cmp=False, f16 = use_f16, mlir_test=False)
        print("compiling ...... ......")
        FakeTensorProp(fx_g).propagate(*example_inputs)
        #fx_pass_for_bmm_expand(fx_g)
        with compilers._disable_jit_autocast():
            compilers.strip_overloads(fx_g) #Remove overloading of node.target, such as aten.sum.dim_intlist to aten.sum]
            tpu_compiler.convert(fx_g)
        _clean_intermidate_file()
        print("compiled done !!!!!!")
    else:
        print("already compiled.")
    module_rt = BmodelModule( bmodel_path, device=example_inputs[0].device, fx_graph = fx_g )
    compiled_fn = module_rt.forward
    return make_boxed_func(compiled_fn)

def _process_None_output(gx_g : torch.fx.GraphModule):
    ''' this func should put in compiler,
        because it's compiler logic.
    '''
    node = None
    for i, n in enumerate(gx_g.graph.nodes):
        if n.op == 'output' : 
            node = n
            break
    args = node.args[0]
    mask = []
    for arg in args:
        if arg :
            mask.append(arg.meta['val'].size())
        else:
            mask.append(None)
    return mask

def tpu_mlir_bwd_compiler(fx_g : torch.fx.GraphModule, example_inputs : List[torch.Tensor] ):
    save_path = f"tmp_bwd"
    use_f16 = False
    dtype = 'f16' if use_f16 else 'f32'
    bmodel_path = f"tmp/{save_path}_{os.environ['CHIP']}_{dtype}_tpu.bmodel"

    compiled = os.path.isfile(bmodel_path)
    if not compiled:
        tpu_compiler = fx2mlir(submodule_name = save_path, chip = os.environ['CHIP'], bwd_graph=True, cmp=False, f16 = use_f16, mlir_test=False)
        print("compiling ...... ......")
        FakeTensorProp(fx_g).propagate(*example_inputs)
        #fx_pass_for_bmm_expand(fx_g)
        with compilers._disable_jit_autocast():
            compilers.strip_overloads(fx_g) #Remove overloading of node.target, such as aten.sum.dim_intlist to aten.sum
            tpu_compiler.convert(fx_g)
        _clean_intermidate_file()
        print("compiled done !!!!!!")
    else:
        print("already compiled.")
    print("start infer ....")
    mask = _process_None_output(fx_g)
    module_rt = BmodelModule( bmodel_path, device=example_inputs[0].device, out_mask = mask,  fx_graph = fx_g )
    compiled_fn = module_rt.forward
    return make_boxed_func(compiled_fn)


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
            # aten.gelu,
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
            aten.mse_loss,
            aten.mse_loss_backward,
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

aot_backend = aot_autograd(bw_compiler =tpu_mlir_bwd_compiler,
                           fw_compiler =tpu_mlir_fwd_compiler,
                           partition_fn=min_cut_rematerialization_partition,
                           decompositions=_get_disc_decomp())
dummy_backend = aot_autograd(bw_compiler =dummy_graph_compiler,
                            fw_compiler = dummy_graph_compiler,
                            partition_fn=min_cut_rematerialization_partition,
                            decompositions=_get_disc_decomp())