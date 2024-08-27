
import torch
import os
import pdb
import torch.nn as nn
import torch_tpu
from torch._functorch.aot_autograd import aot_export_joint_simple, aot_export_module
import torch.optim as optim
from compile.FxGraphConvertor import fx2mlir
import torchvision.models as models
import argparse
import numpy as np
from torch.fx import Interpreter
from torch_tpu.tpu.bmodel_runtime import BmodelRunner, dtype_map, FORMATYPE
from collections import OrderedDict
import json
torch.manual_seed(42)

dump_io = True
device  = torch.device("cpu")

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet50(torch.float32)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        predict = self.model(input)
        loss = self.loss_fn(predict.float(), target.long())
        return loss, predict.detach()

class TraceInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.op_trace = {}
        self.datas = {}
    def run_node(self, n):
        input_names = [i.name if isinstance(i, torch.fx.Node) else i for i in n.args]
        result = super().run_node(n)
        node_name = n.name
        res = {}
        res_name = []
        if isinstance(result, tuple) or isinstance(result, list):
            res = { node_name + f".res{idx}" : value.float().detach().cpu().numpy()  for idx, value in enumerate(result) if value is not None }
            res_name = [node_name + f".res{idx}" for idx in range(len(result))]
        else:
            idx = 0
            if result is not None:
                res[ node_name + f".res{idx}"] = result.float().detach().cpu().numpy()
                res_name = [node_name + f".res{idx}"]

        self.op_trace[n.name] = {"inputs": input_names, "outputs": res_name}
        if "getitem" not in node_name:
            self.datas.update(res)
        return result

def _get_disc_decomp():
    from torch._decomp import get_decompositions
    aten = torch.ops.aten
    decompositions_dict = get_decompositions(
        [
            aten.gelu,
            aten.gelu_backward,
            aten.native_group_norm_backward,
            # aten.native_layer_norm,
            aten.native_layer_norm_backward,
            # aten.std_mean.correction,
            # aten._softmax,
            aten._softmax_backward_data,
            aten.tanh_backward,
            aten.slice_backward,
            aten.select_backward,
            aten.embedding_dense_backward,
            aten.sigmoid_backward,
            aten.nll_loss_backward,
            aten._log_softmax_backward_data,
            aten.nll_loss_forward,
        ]
    )
    return decompositions_dict


def warp_calc(module, idx=0):
    inner_idx = idx
    def forward(*args):
        print(">>>>>>> inputs ")
        for i in range(len(args)):
            print(args[i].abs().max(), args[i].abs().min())
        tinputs = [i.half() if i.dtype == torch.float32 else i for i in args]
        res = module(tinputs)
        print(">>>>>>> outputs ")
        for i in range(len(res)):
            if res[i] is not None:
                print(res[i].abs().max(), res[i].abs().min())
        return res
    return forward

def convert_module_fx(
    submodule_name: str,
    module: torch.fx.GraphModule,
    args={},
    bwd_graph:bool=False,
    para_shape: list=[],
) :
    c = fx2mlir(submodule_name, args, bwd_graph, para_shape)
    return c.convert(module)


def dump_info(kwargs, name="input.json"):
    with open(name, 'w') as f:
        json.dump(kwargs, f)
    # new_ordered_dict = json.loads(json_str, object_pairs_hook = OrderedDict)

def dump_npz(kwargs,name="dump_input.npz"):
    if not dump_io:
        return
    res = {}
    for k in kwargs.keys():
        v = kwargs[k]
        res[k] = v.detach().cpu().numpy()
    np.savez(name, **res)

class SophonJointCompile:

    def __init__(self, model, example_inputs, trace_joint=True, output_loss_index=0, have_32ic=False, args=None):
        fx_g, signature = aot_export_module(
            model, example_inputs, trace_joint=trace_joint, output_loss_index=output_loss_index, decompositions=_get_disc_decomp()
        )
        # signature is ordered dict
        fx_g.to_folder("resnet50_fx","joint")
        self.fx_g                 = fx_g
        self.trace                = TraceInterpreter(fx_g)
        self.signature            = signature
        self.input_maps           = {}
        self.output_maps          = {}
        self.input_maps.update(signature.inputs_to_parameters)
        self.input_maps.update(signature.inputs_to_buffers)
        self.input_maps_reverse   = {v: k for k, v in self.input_maps.items()}
        self.user_inputs          = signature.user_inputs
        self.user_outputs         = signature.user_outputs
        self.output_maps.update(signature.buffers_to_mutate)
        self.output_maps.update(signature.backward_signature.gradients_to_parameters)
        self.output_maps_reverse   = {v: k for k, v in self.output_maps.items()}
        self.info                  = {}
        self.info["input_params"]  = signature.parameters
        self.info["input_buffers"] = signature.buffers
        self.info["output_buffers"]= signature.buffers_to_mutate
        self.info["output_params"] = signature.backward_signature.gradients_to_parameters
        self.info['user_outputs']  = self.user_outputs
        self.info['user_inputs']   = self.user_inputs
        dump_info(self.info, name  = "info.json")
        self.input_num             = len(self.input_maps) + len(self.user_inputs)
        self.output_num            = len(self.output_maps) + len(self.user_outputs)
        self.model                 = model
        self.model_state           = self.model.state_dict()
        self.parameters_keys       = self.signature.parameters
        self.fx_output_list        = list(signature.buffers_to_mutate.keys()) + self.user_outputs + list(signature.backward_signature.gradients_to_parameters.keys())
        self.args                  = args
        self.have_32ic             = have_32ic

    def fx_convert_bmodel(self):
        name = f"test_{args.model}_joint_{args.batch}"
        path = convert_module_fx(name, self.fx_g, self.args, False)
        # path = "./resnet50_8.bmodel"964
        self.bmodel_path = path
        self.model.half()
        self.load_bmodel()

    def check_bmodel_suit(self):
        # check with input and output num
        # check bmodel status
        self.bmodel_name = self.bmodel.model_info["networks"][0]
        self.bmodel_input_info  = self.bmodel.model_net_info[self.bmodel_name]["inputs"]
        self.bmodel_output_info = self.bmodel.model_net_info[self.bmodel_name]["outputs"]
        pass

    def extract_all_input_conv_weight_info(self):
        self.conv_weight       = []
        self.conv_weight_shape = []
        param = self.info['input_params']
        for idx,name in enumerate(param):
            if "weight" in name and len(self.inputs_tpu[idx].shape) == 4:
                self.conv_weight.append(idx)
                self.conv_weight_shape.append(self.bmodel_input_info[idx]["shape"])

    def _prepare_32ic_tensor(self):
        output_weight_start_idx = len(self.info['output_buffers']) + len(self.info['user_outputs'])
        self.tensor_32ic_tpu = {}
        for idx,i in enumerate(self.conv_weight):
            oc, ic, kh, kw = self.conv_weight_shape[idx]
            if kh == kw and kh == 1 and ic % 32 == 0:
                continue
            self.tensor_32ic_tpu[output_weight_start_idx + i] = torch.ops.help.build_tensor([oc, ic, kh, kw], torch.float16, torch.strided, torch.device(f"tpu:0"), False, torch.contiguous_format)

    def init_io_tensors_tpu(self, with_32ic=True):
        self.inputs_tpu = []
        for i in range(self.input_num):
            self.inputs_tpu.append(  self.bmodel.get_model_tensor(i) )
        self.outputs_tpu = []
        for i in range(self.output_num):
            self.outputs_tpu.append( self.bmodel.get_model_tensor(i, is_input=0) )
        self.extract_all_input_conv_weight_info()
        if not with_32ic:
            return
        self._prepare_32ic_tensor()

    def load_bmodel(self):
        # load bmodel
        self.bmodel = BmodelRunner(self.bmodel_path, device_id=0)
        self.check_bmodel_suit()
        self.init_io_tensors_tpu()

    def prepare_model_param_buffer(self):
        total = {}
        for name, param in self.model.named_parameters():
            total[name] = param
        for name, buffer in self.model.named_buffers():
            total[name] = buffer
        return total

    def prepare_model_param_buffer_tpu(self):
        # cpy tensor into tpu
        idx = 0
        for name, param in self.model.named_parameters():
            if str(param.device) != "tpu:0":
                self.inputs_tpu[idx].copy_(param)
                param.data = self.inputs_tpu[idx]
            if param.data_ptr() != self.inputs_tpu[idx].data_ptr():
                print("may be error, please be careful")
            idx += 1
        for name, buffer in self.model.named_buffers():
            if str(buffer.device) != "tpu:0":
                self.inputs_tpu[idx].copy_(buffer.reshape(self.inputs_tpu[idx].shape))
                buffer.data = self.inputs_tpu[idx]
            if buffer.data_ptr() != self.inputs_tpu[idx].data_ptr():
                print("may be error, please be careful")
            idx += 1
        #TODO check all status in device!


    def prepare_model_inputs(self, *args):
        all_inputs = {}
        total = self.prepare_model_param_buffer()
        for k, v in total.items():
            all_inputs[self.input_maps_reverse[k]] = v
        for i, arg in enumerate(args):
            all_inputs[self.user_inputs[i]] = arg
        return all_inputs

    def update_buffers(self, outputs:dict):
        for name, buffer in self.model.named_buffers():
            key = self.output_maps_reverse[name]
            buffer = outputs[key]

    def extract_model_outputs(self, outputs:dict):
        res = []
        for k in self.user_outputs:
            res.append(outputs[k])
        return res

    def call_fx_g(self, *args):
        all_inputs = self.prepare_model_inputs(*args)
        dump_npz(all_inputs, name="resnet50_input.npz")
        if dump_io:
            inputs = list(all_inputs.values())
            all_outputs_list = self.trace.run(*inputs)
        else:
            all_outputs_list = self.fx_g(**all_inputs)
        outputs = { self.fx_output_list[i]: k for i, k in enumerate(all_outputs_list)}
        dump_npz(outputs, name="resnet50_output.npz")
        if dump_io:
            np.savez(f"trace_data_0.npz", **self.trace.datas)
        return outputs

    def extract_model_grads(self, outputs:dict):
        grads = {}
        for key in self.parameters_keys:
            grads[key] = outputs[ self.output_maps_reverse[key] ]
        return grads

    def bind_grad_to_model(self, grads:dict):
        for name, param in self.model.named_parameters():
            param.grad = grads[name]

    def fwdbwd(self, *args):
        outputs = self.call_fx_g(*args)
        self.update_buffers(outputs)
        grads = self.extract_model_grads(outputs)
        self.bind_grad_to_model(grads)
        res = self.extract_model_outputs(outputs)
        return res

    def handle_tpu_args(self, *args):
        idx = len(self.info["input_params"]) + len(self.info["input_buffers"])
        for arg in args:
            self.inputs_tpu[idx].copy_(arg.to(self.inputs_tpu[idx].dtype))
            idx += 1

    def handle_tpu_input(self, *args):
        self.prepare_model_param_buffer_tpu()
        self.handle_tpu_args(*args)

    def call_bmodel(self):
        # use self.inputs_tpu as input and self.outputs_tpu as output
        self.bmodel.forward_sync_with_outputs( self.inputs_tpu, self.outputs_tpu, with_check=False)

    def normalize_outputs(self):
        self.outputs_tpu_fix = [] * len(self.outputs_tpu)
        for i in range(len(self.outputs_tpu)):
            if i in self.tensor_32ic_tpu:
                # TODO add more check
                assert(self.outputs_tpu[i].dtype == torch.float16)
                torch.ops.my_ops.format_cast(self.outputs_tpu[i], self.tensor_32ic_tpu[i], FORMATYPE.Conv_W_32IC_TO_ND)
                self.outputs_tpu_fix.append(self.tensor_32ic_tpu[i])
            else:
                self.outputs_tpu_fix.append(self.outputs_tpu[i])

    def update_buffers_tpu(self):
        idx = 0
        for name, buffer in self.model.named_buffers():
            key = self.output_maps_reverse[name]
            if str(buffer.device) != "tpu:0":
                print("please be careful, maybe error here !!!!!!!!!!!!!!!!!!!!!")
            # tmp = torch.empty(buffer.shape, dtype=buffer.dtype, device=torch.device("tpu:0"))
            # breakpoint()
            # tmp.copy_(self.outputs_tpu[idx])
            # buffer = tmp
            buffer.copy_(self.outputs_tpu[idx].reshape(buffer.shape))
            idx += 1

    def extract_outputs_tpu(self):
        idx = len(self.info["output_buffers"])
        res = []
        for i in range(len(self.user_outputs)):
            res.append(self.outputs_tpu[idx])
            idx += 1
        return res

    def bind_grad_to_model_tpu(self):
        idx = len(self.info["output_buffers"]) + len(self.info["user_outputs"])
        for name, param in self.model.named_parameters():
            param.grad = self.outputs_tpu_fix[idx].reshape(param.shape)
            idx += 1

    def handle_tpu_buffers_params(self):
        self.update_buffers_tpu()
        self.normalize_outputs()
        self.bind_grad_to_model_tpu()
        pass

    def fwdbwdtpu(self, *args):
        self.handle_tpu_input(*args)
        self.call_bmodel()
        self.handle_tpu_buffers_params()
        res = self.extract_outputs_tpu()
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip",  default="bm1690", choices=['bm1684x', 'bm1690'], help="chip name")
    parser.add_argument("--debug", default="", help="debug")
    parser.add_argument("--cmp",   action='store_true', help="enable cmp")
    parser.add_argument("--model", default="resnet50",help="model name")
    parser.add_argument("--fp",    default="fp16",help="fp")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    args = parser.parse_args()
    n = args.batch
    mod = Model()
    mod.train()
    inp = torch.randn((n, 3, 224, 224),dtype = torch.float32)
    target = torch.randint(0, 1000, (n,), dtype=torch.int64)
    joint = SophonJointCompile(mod, [inp, target], trace_joint=True, output_loss_index=0, args=args)
    joint.fx_convert_bmodel()
    opt = optim.SGD(mod.parameters(), lr=0.01, foreach=True)
    for _ in range(10):
        res = joint.fwdbwdtpu(inp, target)
        print("loss ", res[0].cpu().item(), flush=True)
        opt.step()
    # top mlir
    # deploy
    # breakpoint()
    # for _ in range(1):
    #     breakpoint()
    #     res = joint.fwdbwd(inp, target)
    #     print(res[0].item())
    #     opt.step()

    # joint.load_bmodel()
