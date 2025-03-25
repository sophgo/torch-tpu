import torch
import os
import json
import torch.nn as nn
import torch_tpu
import torch.optim as optim
from torch_tpu.tpu.bmodel_runtime import BmodelRunner, dtype_map, FORMATYPE


torch.manual_seed(42)

class TPUWarp:
    def __init__(self, bmodel_path, model, config_path, device_id, chip):
        # config is info json
        self.bmodel_path        = bmodel_path
        self.info               = json.load(open(config_path, 'r'))
        self.model              = model
        self.chip               = chip
        self.device_id          = device_id
        self.bmodel             = BmodelRunner(self.bmodel_path, device_id=device_id)
        self.bmodel_name        = self.bmodel.model_info["networks"][0]
        self.bmodel_input_info  = self.bmodel.model_net_info[self.bmodel_name]["inputs"]
        self.bmodel_output_info = self.bmodel.model_net_info[self.bmodel_name]["outputs"]
        self.input_num          = len(self.bmodel_input_info)
        self.output_num         = len(self.bmodel_output_info)
        self.inputs_tpu         = []
        self.outputs_tpu        = []
        self.extra_input_info   = []
        self.tensor_32ic_tpu    = []
        self.extra_input_num    = 0
        self._build_extra_input()
        self.init_io_tensors_tpu()
        # build self.info from config

    def _build_extra_input(self):
        for i in self.bmodel_input_info:
            if "dropout" in i["name"]:
                self.extra_input_info.append(i)
        self.extra_input_num = len(self.extra_input_info)

    def _prepare_32ic_tensor(self):
        output_weight_start_idx = len(self.info['output_buffers']) + len(self.info['user_outputs'])
        self.tensor_32ic_tpu = {}
        # check with output shape
        for idx,i in enumerate(self.conv_weight):
            oc, ic, kh, kw = self.conv_weight_shape[idx]
            if kh == kw and kh == 1 and ic % 32 == 0:
                continue
            self.tensor_32ic_tpu[output_weight_start_idx + i] = torch.ops.help.build_tensor([oc, ic, kh, kw], torch.float16, torch.strided, torch.device(f"tpu:0"), False, torch.contiguous_format)

    def extract_all_input_conv_weight_info(self):
        self.conv_weight       = []
        self.conv_weight_shape = []
        param = self.info['input_params']
        for idx,name in enumerate(param):
            if "weight" in name and len(self.inputs_tpu[idx].shape) == 4:
                self.conv_weight.append(idx)
                self.conv_weight_shape.append(self.bmodel_input_info[idx]["shape"])

    def init_io_tensors_tpu(self, with_32ic=False):
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

    def prepare_model_param_buffer_tpu(self):
        # cpy tensor into tpu
        idx = 0
        for name, param in self.model.named_parameters():
            if str(param.device) != f"tpu:{self.device_id}":
                self.inputs_tpu[idx].copy_(param)
                param.data = self.inputs_tpu[idx]
            if param.data_ptr() != self.inputs_tpu[idx].data_ptr():
                print("may be error, please be careful")
            idx += 1
        for name, buffer in self.model.named_buffers():
            if str(buffer.device) != f"tpu:{self.device_id}":
                self.inputs_tpu[idx].copy_(buffer.reshape(self.inputs_tpu[idx].shape))
                buffer.data = self.inputs_tpu[idx]
            if buffer.data_ptr() != self.inputs_tpu[idx].data_ptr():
                print("may be error, please be careful")
            idx += 1
        #TODO check all status in device!

    def handle_extra_input_tpu(self):
        if len(self.extra_input_info) == 0:
            return
        idx = len(self.info["input_params"]) + len(self.info["input_buffers"]) + len(self.info["user_inputs"])
        for i in self.extra_input_info:
            if "dropout" in i["name"]:
                shape = i["shape"]
                self.inputs_tpu[idx].copy_( torch.randn(shape) )
                idx += 1
        assert(idx == len(self.inputs_tpu))

    def handle_tpu_args(self, *args):
        idx = len(self.info["input_params"]) + len(self.info["input_buffers"])
        for arg in args:
            self.inputs_tpu[idx].copy_(arg.to(self.inputs_tpu[idx].dtype))
            idx += 1

    def handle_tpu_input(self, *args):
        self.prepare_model_param_buffer_tpu()
        self.handle_tpu_args(*args)
        self.handle_extra_input_tpu()

    def update_buffers_tpu(self):
        idx = 0
        for name, buffer in self.model.named_buffers():
            if str(buffer.device) != "tpu:0":
                print("please be careful, maybe error here !!!!!!!!!!!!!!!!!!!!!")
            buffer.copy_(self.outputs_tpu[idx].reshape(buffer.shape))
            idx += 1

    def normalize_outputs(self):
        self.outputs_tpu_fix = []
        for i in range(len(self.outputs_tpu)):
            if i in self.tensor_32ic_tpu:
                # TODO add more check
                assert(self.outputs_tpu[i].dtype == torch.float16)
                torch.ops.my_ops.format_cast(self.outputs_tpu[i], self.tensor_32ic_tpu[i], 1)
                self.outputs_tpu_fix.append(self.tensor_32ic_tpu[i])
            else:
                self.outputs_tpu_fix.append(self.outputs_tpu[i])

    def bind_grad_to_model_tpu(self):
        idx = len(self.info["output_buffers"]) + len(self.info["user_outputs"])
        for name, param in self.model.named_parameters():
            param.grad = self.outputs_tpu_fix[idx].reshape(param.shape)
            idx += 1

    def handle_tpu_buffers_params(self):
        self.update_buffers_tpu()
        self.normalize_outputs()
        self.bind_grad_to_model_tpu()

    def call_bmodel(self):
        self.bmodel.forward_sync_with_outputs( self.inputs_tpu, self.outputs_tpu, with_check=False)

    def extract_outputs_tpu(self):
        idx = len(self.info["output_buffers"])
        res = []
        for i in range(len(self.info['user_outputs'])):
            res.append(self.outputs_tpu[idx])
            res[-1].requires_grad = True
            idx += 1
        return res

    def forward(self, *args):
        self.handle_tpu_input(*args)
        self.call_bmodel()
        self.handle_tpu_buffers_params()
        res = self.extract_outputs_tpu()
        return res

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def named_buffers(self):
        return self.model.named_buffers()

    def state_dict(self):
        return self.model.state_dict()

def TPUCompile(bmodel_path, config_path, device_id=0, chip="bm1684x"):
    def decorator(cls):
        def wrapper(*args, **kwargs):
            model = cls(*args, **kwargs)
            model = model.train()
            return TPUWarp(bmodel_path, model, config_path, device_id, chip)
        return wrapper
    return decorator