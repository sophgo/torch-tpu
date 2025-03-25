import torch_tpu
import torch

# only support SG2260

dtype_map = {
    0: torch.float32,
    1: torch.float16,
    2: torch.int8,
    3: torch.uint8,
    4: torch.int16,
    # 5: torch.uint16,
    6: torch.int32,
    # 7: torch.uint32,
    8: torch.bfloat16,
    # 8: torch.int4,
    # 9: torch.uint4,
}

class FORMATYPE:
    Conv_W_ND_TO_32IC     = 0
    Conv_W_32IC_TO_ND     = 1
    Conv_W_ND_TO_32IC32OC = 2
    Conv_W_32IC32OC_TO_ND = 3
    Conv_DW_32OC_TO_ND    = 4
    Conv_DW_ND_TO_32OC    = 5

class BmodelRunner:
    # TODO : support multi stages and dynamic shape
    def __init__(self, model_path, device_id=0, decrypt_lib=""):
        """Initialize the BmodelRun class with load the model.
        """
        self.model_path  = model_path
        self.device_id   = device_id
        self.decrypt_lib = decrypt_lib
        self.load_model()
        self.load_model_net_info()
        self.cur_net      = None
        self.cur_net_name = None

    def __del__(self):
        print("BmodelRunner delete")
        del self.nets
        del self.model
        # import gc; gc.collect()

    def load_model(self):
        self.model      = torch_tpu._C.Model(self.model_path, self.device_id, self.decrypt_lib)
        self.model_info = torch_tpu._C.getModelInfo(self.model)

    def load_model_net_info(self):
        self.nets           = {}
        self.model_net_info = {}
        for net_name in self.model_info["networks"]:
            self.nets[net_name]           = torch_tpu._C.Net(self.model, net_name)
            self.model_net_info[net_name] = torch_tpu._C.getNetworkInfo(self.nets[net_name])

    def _check_cur_net(self):
        if self.cur_net is None:
            self.set_cur_net()

    def get_model_tensor(self, idx, is_input=1):
        self._check_cur_net()
        assert self.cur_net is not None, "Please set the current net first"
        assert 0 <= idx and idx < self.model_net_info[self.cur_net_name]["num_input"] if is_input else self.model_net_info[self.cur_net_name]["num_output"], "The idx is out of range"
        return torch_tpu._C.getTensor(self.cur_net, idx, is_input)

    def prepare_outputs(self, build_in=False):
        """Prepare the output tensor for the model.
        Parameters:
            build_in: bool, whether the output tensor is build in the model (use empty or torch.ops.help.build_tensor).
        """
        self._check_cur_net()
        outputs = []
        for i in range(self.model_net_info[self.cur_net_name]["num_output"]):
            shape = self.model_net_info[self.cur_net_name]["outputs"][i]["shape"]
            dtype = self.model_net_info[self.cur_net_name]["outputs"][i]["dtype"]
            dtype = dtype_map[dtype]
            if build_in:
                outputs.append(torch.ops.help.build_tensor(shape, dtype, torch.strided, torch.device(f"tpu:{self.device_id}"), False, torch.contiguous_format))
            else:
                outputs.append(torch.empty(shape, dtype=dtype, device=torch.device(f"tpu:{self.device_id}")))
        return outputs

    def forward_sync(self, inputs):
        self._check_cur_net()
        outputs = self.prepare_outputs()
        self.forward_sync_with_outputs(inputs, outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        self._check_cur_net()
        outputs = self.prepare_outputs()
        self.forward_with_outputs(inputs, outputs)
        return outputs if len(outputs) > 1 else outputs[0]

    def check_input_outputs(self, inputs, outputs):
        """check the inputs and outputs tensor shape and dtype.
        Parameters:
            inputs:  list, input tensor
            outputs: list, output tensor
        """
        net_info = self.model_net_info[self.cur_net_name]
        # level 0 check the input and output tensor number
        if len(inputs) != net_info["num_input"]:
            raise ValueError("The input tensor number is not equal to the model input number")
        if len(outputs) != net_info["num_output"]:
            raise ValueError("The output tensor number is not equal to the model output number")
        # level 1 check the input and output tensor in inneed device
        for i in range(net_info["num_input"]):
            if inputs[i].device != torch.device(f"tpu:{self.device_id}"):
                raise ValueError(f"The tensor {i} is not in the device {self.device_id}")
        for i in range(net_info["num_output"]):
            if outputs[i].device != torch.device(f"tpu:{self.device_id}"):
                raise ValueError(f"The tensor {i} is not in the device {self.device_id}")
        # level 2 check the input and output tensor shape dtype size
        # for i in range(net_info["num_input"]):
        #     if list(inputs[i].shape) != net_info['inputs'][i]["shape"]:
        #         raise ValueError(f"The input tensor {i} shape is not equal to the model input shape")
        #     if inputs[i].dtype != dtype_map[net_info['inputs'][i]["dtype"]]:
        #         raise ValueError(f"The input tensor {i} dtype is not equal to the model input dtype")
        # for i in range(net_info["num_output"]):
        #     if list(outputs[i].shape) != net_info['outputs'][i]["shape"]:
        #         raise ValueError(f"The output tensor {i} shape is not equal to the model output shape")
        #     if outputs[i].dtype != dtype_map[net_info['outputs'][i]["dtype"]]:
        #         raise ValueError(f"The output tensor {i} dtype is not equal to the model output dtype")

    def set_cur_net(self, net_name=None):
        if net_name is None:
            self.cur_net_name = self.model_info["networks"][0]
        else:
            if net_name not in self.model_info["networks"]:
                raise ValueError("Net name not in the model networks")
            self.cur_net_name = net_name
        self.cur_net = self.nets[self.cur_net_name]

    def forward_with_outputs(self, inputs, outputs, with_check=False):
        """Run the model with inputs and outputs.
        Parameters:
            inputs:  list, input tensor
            outputs: list, output tensor
        """
        if isinstance(inputs, list):
            pass
        elif not isinstance(inputs, list) and isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        else:
            assert 0, "input should be instance of list or instance of Tensor"
        assert isinstance(outputs, list), "outputs should be a instance of list"

        if with_check:
            self.check_input_outputs(inputs, outputs)
        torch_tpu._C.forward(self.cur_net, inputs, outputs)

    def forward_sync_with_outputs(self, inputs, outputs, with_check=False):
        """Run the model with inputs and outputs.
        Parameters:
            inputs:  list, input tensor
            outputs: list, output tensor
        """
        if not isinstance(inputs, list) and isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        elif isinstance(inputs, list):
            pass
        else:
            assert 0, "input should be instance of list or instance of Tensor"
        assert isinstance(outputs, list), "outputs should be a instance of list"

        if with_check:
            self.check_input_outputs(inputs, outputs)
        torch_tpu._C.forward_sync(self.cur_net, inputs, outputs)