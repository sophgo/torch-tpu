import torch
import torch_tpu
import torch.nn as nn
import copy
from torch_tpu.dynamo import aot_backend, dummy_backend

def test_silu():
    device = 'tpu'
    #              inp_shape,     ic,   inp_grad, bias_grad 
    test_cases = [((8,32,320,320), 32,      True,     True),
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

    net = nn.SiLU()
    net_tpu = copy.deepcopy(net).to(device).train()
    net_opt = torch.compile(net_tpu, backend=aot_backend, dynamic=None, fullgraph=False)

    out = net(inp)

    #out_tpu = net_tpu(inp_tpu) # eager mode
    out_tpu = net_opt(inp_tpu)  # compile mode

    diff_o = out - out_tpu.cpu()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_silu()
