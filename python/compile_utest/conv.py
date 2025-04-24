import torch
import torch_tpu
import torch.nn as nn
import copy
from torch_tpu.dynamo import aot_backend, dummy_backend

def test_conv():
    device = 'tpu'
    #              inp_shape,     ic,   oc,   k,   pad, stride, inp_grad, bias_grad 
    test_cases = [((8,3,640,640), 3,    32,   6,   2,   2,      True,     True),
                  ((8,256,40,40), 256,  256,  1,   0,   1,      True,     False),  
                  ]

    cs = test_cases[0]
    inp_shape    = cs[0]
    ic           = cs[1]
    oc           = cs[2]
    kernel_s     = cs[3]
    padding      = cs[4]
    stride       = cs[5]
    inp_req_grad = cs[6]
    bias_req_grad= cs[7]

    inp = torch.randn(inp_shape)
    inp_tpu = copy.deepcopy(inp).to(device)
    inp.requires_grad     = inp_req_grad
    inp_tpu.requires_grad = inp_req_grad

    net = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=kernel_s, padding=padding,stride=stride)
    net_tpu = copy.deepcopy(net).to(device)

    net.bias.requires_grad     = bias_req_grad
    net_tpu.bias.requires_grad = bias_req_grad

    net_opt = torch.compile(net_tpu, backend=aot_backend, dynamic=None, fullgraph=False)

    out = net(inp)

    #out_tpu = net_tpu(inp_tpu)
    out_tpu = net_opt(inp_tpu)

    grad_o = torch.rand_like(out)
    grad_o_cpu = grad_o.to(device)

    out.backward(grad_o)
    out_tpu.backward(grad_o_cpu)

    w_g_diff = net.weight.grad - net_tpu.weight.grad.cpu()

    if bias_req_grad:
        b_g_diff = net.bias.grad   - net_tpu.bias.grad.cpu()
    
    if inp_req_grad:
        i_g_diff = inp.grad        - inp_tpu.grad.cpu()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_conv()
