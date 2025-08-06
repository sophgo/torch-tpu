import torch
import torch_tpu
import torch.nn as nn
import copy
from torch_tpu.dynamo import aot_backend, dummy_backend

def test_conv():
    device = 'tpu'
    #              inp_shape,     ic,   oc,   k,   pad, stride, inp_grad, bias_grad 
    test_cases = [((8,3,640,640), 3,    32,   6,   2,   2,      True,     False),
                  ((8,256,40,40), 256,  255,  1,   0,   1,      True,     True), # Error: w, bias
                  ((8,128,80,80), 128,  255,  1,   0,   1,      True,     True), # ERROR: bias
                  ((8,512,20,20), 512,  255,  1,   0,   1,      True,     True),
                  ]
    for i in range(len(test_cases)):
        cs = test_cases[1]
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

        grad_o = torch.randn_like(out)
        grad_o_cpu = grad_o.to(device)

        out.backward(grad_o)
        out_tpu.backward(grad_o_cpu)

        w_g_diff = abs(net.weight.grad - net_tpu.weight.grad.cpu())
        print(f"w-grad max-diff = {torch.max(w_g_diff)}")

        if bias_req_grad:
            b_g_diff = abs(net.bias.grad   - net_tpu.bias.grad.cpu())
            print(f"bias-grad max-diff = {torch.max(b_g_diff)}")
        if inp_req_grad:
            i_g_diff = abs(inp.grad        - inp_tpu.grad.cpu())
            print(f"in-grad max-diff = {torch.max(i_g_diff)}")

        import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_conv()