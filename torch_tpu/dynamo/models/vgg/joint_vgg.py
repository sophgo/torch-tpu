
import torch
import os
import pdb
import torch_tpu
import json
import torch.nn as nn
from torch_tpu.tpu.bmodel_runtime import BmodelRunner, dtype_map, FORMATYPE
from collections import OrderedDict
torch.manual_seed(42)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet50(torch.float32)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        predict = self.model(input)
        loss = self.loss_fn(predict.float(), target.long())
        return loss, predict.detach()

# build a fake module from config
# 今晚写完这个 应该可以
# 如果是一个模型 就直接传进去；如果是一个config 就需要先构建一个模型。
# 这一块的处理 会有点麻烦.还得想着libtorch怎么处理

class BmodelTrain:

    def __init__(self, model, model_path, config, device_id, chip, io_config=None ):
        self.model_path         = model_path
        self.config             = config
        self.model              = model
        self.chip               = chip
        self.device_id          = device_id
        self.bmodel             = BmodelRunner(self.model_path, device_id=device_id)
        self.bmodel_name        = self.bmodel.model_info["networks"][0]
        self.bmodel_input_info  = self.bmodel.model_net_info[self.bmodel_name]["inputs"]
        self.bmodel_output_info = self.bmodel.model_net_info[self.bmodel_name]["outputs"]
        breakpoint()
        self.inputs_tpu         = []
        self.outputs_tpu        = []

        self.io_config          = io_config
        # extract bmodel inputs
        # self.model需要是 nn.Module的子类 想想办法
        # 需要处理self.info self.inputs 和 self.outputs

    def _basic_io_tensors_tpu(self):
        self.inputs_tpu = []
        self.outputs_tpu = []
        for i in range(len(self.bmodel_input_info)):
            self.inputs_tpu.append(
                self.bmodel.get_model_tensor(i)
            )
        for i in range(len(self.bmodel_output_info)):
            self.outputs_tpu.append(
                self.bmodel.get_model_tensor(i, is_input=0)
            )

    def _prepare_other_tensors_tpu(self):
        pass


    def init_io_tensors_tpu(self):
        self._basic_io_tensors_tpu()

        pass

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

    def handle_dropout_input_tpu(self):
        # extract bmodel input
        pass

    def handle_extra_input_tpu(self):
        # extract bmodel input
        #    dropout
        #    xxxxxx (will add later)
        self.handle_dropout_input_tpu()
        pass

    def handle_tpu_input(self, *args):
        self.prepare_model_param_buffer_tpu()
        self.handle_tpu_args(*args)
        self.handle_extra_input_tpu()

    def update_buffers_tpu(self):
        idx = 0
        for name, buffer in self.model.named_buffers():
            key = self.output_maps_reverse[name]
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

    def handle_tpu_buffers_params(self):
        self.update_buffers_tpu()
        self.normalize_outputs()
        self.bind_grad_to_model_tpu()

    def bind_grad_to_model_tpu(self):
        idx = len(self.info["output_buffers"]) + len(self.info["user_outputs"])
        for name, param in self.model.named_parameters():
            param.grad = self.outputs_tpu_fix[idx].reshape(param.shape)
            idx += 1

    def call_bmodel(self):
        self.bmodel.forward_sync_with_outputs( self.inputs_tpu, self.outputs_tpu, with_check=False)

    def extract_outputs_tpu(self):
        idx = len(self.info["output_buffers"])
        res = []
        for i in range(len(self.user_outputs)):
            res.append(self.outputs_tpu[idx])
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


model_path = "/workspace/newcompile/models2/vgg/vgg_8.bmodel"
# load json
config = json.load(open("info.json",'r'))

btrain = BmodelTrain(None, model_path, config, 0, "bm1690")
breakpoint()