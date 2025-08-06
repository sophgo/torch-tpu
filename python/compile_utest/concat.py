import torch
import torch_tpu
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_tpu.dynamo import aot_backend, dummy_backend

class Concat(nn.Module):
    """Concatenates tensors along a specified dimension for efficient tensor manipulation in neural networks."""

    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)

def test_cat():
    device = 'tpu'
    #              inp_shape,     ic,   inp_grad, bias_grad 
    test_cases = [((1,3,2,2),  32,      True,     True),
                  ]

    cs = test_cases[0]
    inp_shape    = cs[0]
    ic           = cs[1]
    inp_req_grad = cs[2]
    bias_req_grad= cs[3]

    inp = torch.rand(inp_shape)
    inp2= torch.rand(inp_shape)
    inp_tpu = copy.deepcopy(inp).to(device)
    inp_tpu2 = copy.deepcopy(inp2).to(device)

    inp.requires_grad     = inp_req_grad
    inp_tpu.requires_grad = inp_req_grad
    inp2.requires_grad     = inp_req_grad
    inp_tpu2.requires_grad = inp_req_grad

    net = Concat()
    net_tpu = copy.deepcopy(net).to(device).train()
    net_opt = torch.compile(net_tpu, backend=aot_backend, dynamic=None, fullgraph=False)
    out = net((inp,inp2))
    import pdb; pdb.set_trace()
    #out_tpu = net_tpu((inp_tpu,inp_tpu2)) # eager mode
    out_tpu = net_opt((inp_tpu,inp_tpu2))  # compile mode
    import pdb; pdb.set_trace()

    grad_o = torch.range(1, out.numel()).view(out.shape).to(out.dtype)
    grad_o_tpu = grad_o.to(device)

    out.backward(grad_o)
    out_tpu.backward(grad_o_tpu)

    diff_o = out - out_tpu.cpu()
    diff_i = inp.grad-inp_tpu.grad.cpu()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_cat()
