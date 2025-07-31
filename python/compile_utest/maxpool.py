import torch
import torch_tpu
import torch.nn as nn
import copy
from torch_tpu.dynamo import aot_backend, dummy_backend

def test_bn_fwd():
    device = 'tpu'
    #              inp_shape,     ic,  k, pad,  inp_grad, bias_grad 
    test_cases = [((8,256,20,20), 256, 5,   2,      True,     True),
                  ]

    cs = test_cases[0]
    inp_shape    = cs[0]
    ic           = cs[1]
    k            = cs[2]
    pad          = cs[3]
    inp_req_grad = cs[4]
    bias_req_grad= cs[5]

    inp = torch.randn(inp_shape)
    inp_tpu = copy.deepcopy(inp).to(device)
    inp.requires_grad     = inp_req_grad
    inp_tpu.requires_grad = inp_req_grad

    net = nn.MaxPool2d(kernel_size=k, stride=1, padding=pad, return_indices=True)
    net_tpu = copy.deepcopy(net).to(device).train()
    net_opt = torch.compile(net_tpu, backend=aot_backend, dynamic=None, fullgraph=False)

    out, o_i = net(inp)
    out_tpu, o_i_tpu = net_tpu(inp_tpu) # eager mode
    # out_tpu, o_i_tpu = net_opt(inp_tpu)  # compile mode

    diff_o   = abs(out - out_tpu.cpu())
    diff_o_i = abs(o_i - o_i_tpu.cpu())
    print(f"val : {torch.max(diff_o)}")
    print(f"ind : {torch.max(diff_o_i)}")
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_bn_fwd()