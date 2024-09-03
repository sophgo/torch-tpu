import os
import torch
import pdb
import gc
import time
import copy
import numpy as np
import logging
from tpu_mlir.python import pyruntime_bm
from tpu_mlir.python.numpy_helper.npz_compare import npz_compare
tpu_dev = "privateuseone:0"
device = torch.device(tpu_dev)

def torch_dtype_from_tpu_mlir(dtype) -> torch.dtype:
    if dtype == 'f16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    elif dtype == 'f32':
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TpuMlirModule(torch.nn.Module):
    def __init__(
        self, args, model_file, in_tensor_name_to_idx_dict, output_changed_shapes,\
        output_tensor_names = None, output_dtypes = None, return_none_count = 0,\
        in_ref_data = {}
    ):
        super(TpuMlirModule, self).__init__()
        print(f'TpuMlirModule __init__ output_dtypes:{output_dtypes}')
        self._register_state_dict_hook(TpuMlirModule._on_state_dict)
        self.args = args
        self.output_dtypes = output_dtypes
        self.model_file = model_file
        self.initialized = False
        self.return_none_count = return_none_count
        self.output_changed_shapes = output_changed_shapes
        self.in_tensor_name_to_idx_dict = in_tensor_name_to_idx_dict
        self.output_tensor_names = output_tensor_names
        self.output_tensor_names = None
        self.in_ref_data = in_ref_data

        if model_file:
            self._initialize()

    def _initialize(self):
        print('_initialize for', self.args.chip)
        # os.system('ln -sf $TPUC_ROOT/lib/libcmodel_1684x.so $TPUC_ROOT/lib/libcmodel.so')
        self.model = pyruntime_bm.Model(self.model_file)
        self.net = self.model.Net(self.model.networks[0])
        self.initialized = True

    def engineToBmodel(self):
        with open(self.model_file, "wb") as fd:
            fd.write(self.engine)

    def _check_initialized(self):
        if not self.initialized:
            raise RuntimeError("TpuMlirModule is not initialized.")

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        self._check_initialized()
        with open(self.model_file, 'rb') as fd:
            state_dict[prefix + "engine"] = bytearray(fd.read())

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.engine = state_dict[prefix + "engine"]
        self._initialize()

    def __getstate__(self):
        state = self.__dict__.copy()
        with open(self.model_file, "rb") as fd:
            state["engine"] = bytearray(fd.read())
        return state

    def __setstate__(self, state):
        self.engineToBmodel()
        self.__dict__.update(state)

    def forward(self, *inputs):
        print(f'>>>runtime call bmodel:{self.model_file}:')
        print('input info:')
        new_inputs = []
        for net_input in self.net.inputs:
            torch_input = inputs[self.in_tensor_name_to_idx_dict[net_input.name]]
            if list(torch_input.shape) != list(net_input.data.shape):
                torch_input = torch_input.reshape(tuple(net_input.data.shape))
            new_inputs.append(torch_input)
            print(f' bmodel input:{net_input.name} shape:{net_input.data.shape}, torch input shape:{torch_input.shape}')

        with torch.autograd.profiler.record_function("TpuMlirModule:Forward"):
            self._check_initialized()
            input_shapes = []
            with torch.autograd.profiler.record_function("TpuMlirModule:ProcessInputs"):
                contiguous_inputs = [i.contiguous() if isinstance(i, torch.Tensor) else i for i in new_inputs]
                i = 0
                for net_input in self.net.inputs:
                    # assert contiguous_inputs[
                    #     i
                    # ].is_privateuseone, f"{i}th input({net_input.name}) is not on tpu device."

                    # dtype = torch_dtype_from_tpu_mlir(net_input.data.dtype)
                    # assert (
                    #     contiguous_inputs[i].dtype == dtype
                    # ), f"Dtype mismatch for {i}th input({net_input.name}). Expect {dtype}, got {contiguous_inputs[i].dtype}."
                    input = contiguous_inputs[i]
                    input = input if isinstance(input, np.ndarray) else input.cpu().numpy()
                    input_shapes.append(input.shape)
                    if len(input.shape) == 0 or list(input.shape) == [1]:
                        net_input.data = input
                    else:
                        net_input.data[:] = input
                    i += 1

            dyn = False
            with torch.autograd.profiler.record_function("TpuMlirModule:TpuRuntime"):
                if dyn:
                    dyn_output_shapes = self.net.forward_dynamic(input_shapes)
                else:
                    t0 = time.time()
                    dyn_output_shapes = self.net.forward()
                    print(f'time:{time.time()-t0}')
            with torch.autograd.profiler.record_function("TpuMlirModule:ProcessOutputs"):
                # create output tensors
                tpu_outputs: List[torch.Tensor] = []

                dyn_idx = 0
                output_dict = {}
                for i in self.net.outputs:
                    if self.output_tensor_names is not None and i.name not in self.output_tensor_names:
                        print('skip:', i.name)
                        continue
                    output = np.array(i.data)
                    output_dict[i.name] = np.array(i.data.astype(np.float32))
                    if dyn:
                        if output.shape != dyn_output_shapes[dyn_idx]:
                            dyn_len = np.prod(dyn_output_shapes[dyn_idx])
                            output = output.flatten()[:dyn_len].reshape(
                                *dyn_output_shapes[dyn_idx])
                            dyn_idx += 1
                    tmp = torch.from_numpy(output)
                    if i.name in self.output_changed_shapes:
                        tmp = tmp.reshape(self.output_changed_shapes[i.name])
                        print(f'{i.name} reshape from {i.data.shape} to {self.output_changed_shapes[i.name]}')
                    if tmp.shape == torch.Size([1]):
                        tmp = tmp[0]
                    tpu_outputs.append(tmp)
                if self.output_dtypes is not None:
                    for output, dtype in zip(tpu_outputs, self.output_dtypes):
                        if dtype == torch.int64:
                            output = output.int()
                print('forward output shape:', [i.shape for i in tpu_outputs])

                ### bmodel out compare ###
                if self.args.cmp:
                    import pdb;pdb.set_trace()
                    np.savez('bmodel_out_data.npz', **output_dict)
                    del output_dict
                    gc.collect()
                    npz_compare(['bmodel_out_data.npz', 'ref_data.npz', "--tolerance", "0.99,0.98", "-v"])
                
                if self.return_none_count > 0:
                    tpu_outputs.extend([None for i in range(self.return_none_count)])
                    print('return_none_count:', self.return_none_count)
            if len(tpu_outputs) == 1:
                return tpu_outputs[0]
            for i in range(len(tpu_outputs)):
                if tpu_outputs[i]!=None:
                    tpu_outputs[i] = tpu_outputs[i].to(device)
            return tuple(tpu_outputs)

    def get_layer_info(self) -> str:
        """
        Get layer info of the engine. Only support for TRT > 8.2.
        """
        # inspector = self.engine.create_engine_inspector()
        # return inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        pass
