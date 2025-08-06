import torch
import torch_tpu
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_tpu.dynamo import aot_backend, dummy_backend

def test_upsample():
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
    inp_tpu = copy.deepcopy(inp).to(device)
    inp.requires_grad     = inp_req_grad
    inp_tpu.requires_grad = inp_req_grad

    net = nn.Upsample(scale_factor=2.0)
    net_tpu = copy.deepcopy(net).to(device).train()
    net_opt = torch.compile(net_tpu, backend=aot_backend, dynamic=None, fullgraph=False)
    out = net(inp)
    import pdb; pdb.set_trace()
    #out_tpu = net_tpu(inp_tpu) # eager mode
    out_tpu = net_opt(inp_tpu)  # compile mode
    import pdb; pdb.set_trace()

    grad_o = torch.range(1, out.numel()).view(out.shape).to(out.dtype)
    grad_o_tpu = grad_o.to(device)
    import pdb; pdb.set_trace()

    out.backward(grad_o)
    out_tpu.backward(grad_o_tpu)
    import pdb; pdb.set_trace()

    diff_o = out - out_tpu.cpu()
    diff_i = inp.grad-inp_tpu.grad.cpu()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_upsample()
