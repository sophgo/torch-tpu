import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    found_inf = torch.zeros([1]).to(device).half()
    inv_scale = torch.ones([1]).to(device) * 70000
    grads = [torch.ones([320, 240]).to(device).half(), torch.ones([320, 240]).to(device).half()* 70000 ]
    torch._amp_foreach_non_finite_check_and_unscale_(
            grads,
            found_inf,
            inv_scale)
    print("found_inf: ", found_inf.cpu())
    print("inv_scale: ", inv_scale.cpu())
    for i in range(len(grads)):
       print("grad: ", grads[i].cpu()[:10])
    import pdb;pdb.set_trace()
    
if __name__ == "__main__":
    case1()