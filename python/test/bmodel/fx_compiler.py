import torch
import torch_tpu
from torch_tpu.dynamo import aot_backend, dummy_backend

device = 'tpu:0'

def func(w, x):
    o = torch.mm(w, x)
    return o

class module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w =  torch.nn.parameter.Parameter(torch.ones((32,16), device=device))
    def forward(self, x):
        return torch.mm(self.w, x)

def case1():
    """ dummy_backend
        will save graph_table.txt
    """
    input   = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    o_ori   = func(weight, input)

    opt_func=torch.compile(func, backend=dummy_backend)
    o_opt = opt_func(weight, input)

    diff = abs(o_ori-o_opt)
    print(torch.max(diff))

def case2():
    """ aot_backend
    """
    input   = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    o_ori   = func(weight, input)

    opt_func=torch.compile(func, backend=aot_backend)
    o_opt = opt_func(weight, input)

    diff = abs(o_ori-o_opt)
    print(torch.max(diff))

def case3():
    """fw + bw
    """
    grad_   = torch.ones((32, 34), device=device)
    input   = torch.ones((16, 34), device=device)
    model   = module()
    o_ori   = model(input)
    o_ori.backward(grad_)
    import pdb; pdb.set_trace()

    opt_model = torch.compile(model, backend=aot_backend, )
    o_opt     = opt_model(input)
    import pdb; pdb.set_trace()
    o_opt.backward(grad_)
    import pdb; pdb.set_trace()


def case4():
    """ fw + loss + bw
    """
    grad_   = torch.ones((32, 34), device=device)
    input   = torch.ones((16, 34), device=device)
    #input.requires_grad = True
    model   = module()
    o_ori   = model(input)
    loss    = torch.sum(o_ori - grad_)
    loss.backward()

    opt_model = torch.compile(model, backend=aot_backend, dynamic = None, fullgraph=False)
    o_opt     = opt_model(input)

    loss_opt    = torch.sum(o_opt - grad_)
    loss_opt.backward()
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    # case1()
    # case2()
    case3()
    # case4()