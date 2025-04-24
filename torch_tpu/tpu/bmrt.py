import torch
import torch_tpu
import torch_tpu._C

import os

def check_tpu_kernel_module():
    kernel_module = os.path.join(os.path.expanduser("~"),'.torch_tpu_kernel_module')
    if os.path.isfile(kernel_module):
        UserWarning(f"use {kernel_module}")
        UserWarning('if user not make sure kernel_module is good,  `export TorchTpuSaveKernelModule=1` generate new one')
    else:
        raise RuntimeError(f"{kernel_module} not exist, please `export TorchTpuSaveKernelModule=1` generate fisrtly")

class BmodelRuner(torch_tpu._C._TPUBModelRtBase):
    r"""Wrapper around a BmodelRuner.

        A BmodelRuner is a wrapper of modelrt. which can run bmodel directly.
    
        warning: 
            this features must run with environ 'ModelRtRunWithTorchTpu=1'
        Args:
    """
    def __new__(cls, device_id=0, stream_id=0, bmodel_path="", decrypt_lib=""):
        os.environ['ModelRtRunWithTorchTpu'] = '1'
        check_tpu_kernel_module()
        with torch_tpu.tpu.device(device_id):
            return super(BmodelRuner, cls).__new__(cls, stream_id=stream_id,
                        device_index=device_id, bmodel_path=bmodel_path, decrypt_lib=decrypt_lib)
        
    def __init__(self, device_id=0, stream_id=0, bmodel_path="", decrypt_lib=""):
       (self.input_names, self.output_names) = super(BmodelRuner, self).GetIONames()

    def get_io_names(self):
        return self.input_names, self.output_names

    def forward(self, x, out, non_blocking=False):
        super(BmodelRuner, self).forward(x, out, non_blocking)

    def set_runingnet(self, net_name=""):
        super(BmodelRuner, self).SetRuningNet(net_name)
    
    def genInplaceIO(self):
        (inputs, outputs) = super(BmodelRuner, self).GenInplaceTensor()
        return inputs, outputs
        
class BmodelModule(torch.nn.Module):
    def __init__(self,  bmodel_path='', device=torch.device('tpu:0'), out_mask = None, fx_graph=None):
        super(BmodelModule, self).__init__()
        device_id = int(str(device).split(":")[1]) if ':' in str(device) else 0
        self.outmask = out_mask
        self.bmodel_path = bmodel_path
        self.bmrt    = BmodelRuner(device_id= device_id, bmodel_path=bmodel_path)
        self.inputs, self.outputs = self.bmrt.genInplaceIO()
        self.fx_g = fx_graph
        self.fx_inp_names = []
        self.fx_out_names = []
        if self.fx_g is not None:
            self.get_io_name_idx()
        self.save_io = os.environ.get('ModelRtWTorchDEBUG')

    def get_io_name_idx(self,):
        assert self.fx_g is not None
        for i, node in enumerate(self.fx_g.graph.nodes):
            if node.op == 'placeholder':
                self.fx_inp_names.append(node.name)
        output_node = [i for i in self.fx_g.graph.nodes if i.op == 'output' and len(i.args) > 0][0]
        for node in output_node.args[0]:
            if node is None: continue
            if '_to_copy' in node.name or 'clone' in node.name: ## this is should not be here, we should change compiler's code
                pre_node = node.args[0]
                self.fx_out_names.append(pre_node.name)
            else:
                self.fx_out_names.append(node.name)

    def get_inplace_io(self):
        """return inplace io of module runtime
            user can use this function
                1. get inputs, as pre-op's out.
                2. get outputs, as next-op's input
            which like a graph create, will better for performance. 
        """
        return (self.inputs, self.outputs)

    def post_process_output(self, default_dtype=torch.float32):
        """post processing enusres the output aligns with the framework's input format.
        """
        # 1.dtype convert
        outs_dtype = []
        for o in self.outputs:
            if o.dtype != default_dtype:
                outs_dtype.append(o.to(default_dtype))
            else:
                outs_dtype.append(o)

        # 2.order ajust, according to names
        outs_ = []
        for fx_o_name in self.fx_out_names:
            b_idx = None
            for idx, bmodel_o_name in enumerate(self.bmrt.output_names):
                if fx_o_name == bmodel_o_name or fx_o_name + "_f32" == bmodel_o_name or fx_o_name + "_folder" == bmodel_o_name:
                    b_idx = idx
                    break
            assert b_idx is not None, f"{fx_o_name} not fond in bmodel"

            outs_.append(outs_dtype[b_idx])

        # 3. process None outputs
        if self.outmask is None: return outs_
        else:
            outs = []
            idx_out = 0
            for size in self.outmask:
                if size:
                    outs.append(outs_[idx_out].view(size))
                    idx_out += 1
                else:
                    outs.append(None)
            return outs

    def forward(self, *inputs):
        assert isinstance(inputs, tuple)
        for idx, input in enumerate(inputs):
            if self.inputs[idx].data_ptr() == input.data_ptr():
                continue
            else:
                if self.inputs[idx].dtype == input.dtype:
                    self.inputs[idx].copy_(input)
                else:
                    input_d = input.to(self.inputs[idx].dtype)
                    self.inputs[idx].copy_(input_d)
                    ### need to refine to save once dma operation
                    #torch.ops.aten._copy_from(input, self.inputs[idx], False)
        self.bmrt.forward(self.inputs, self.outputs)

        outs = self.post_process_output(inputs[0].dtype)
        if self.save_io:
            i_path = self.bmodel_path.split(".bmodel")[0] + "_i.tensor"
            o_path = self.bmodel_path.split(".bmodel")[0] + "_o.tensor"
            i_cpus = [i_.cpu() for i_ in inputs]
            torch.save(i_cpus, i_path)
            o_cpus = [o_.cpu() if o_ is not None else None for o_ in outs]
            torch.save(o_cpus,   o_path)
        return outs