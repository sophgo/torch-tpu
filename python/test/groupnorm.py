import torch
import torch.nn as nn
import copy
import torch_tpu
import torch.nn.functional as F
import time
torch.manual_seed(1000)

device = "tpu"
N = 1
C = 8
H = 1
W = 2
NUM_GROUPS = 4
eps = 1e-9


def simulate_forward(x, net):
    w = net.weight
    bias = net.bias
    x_o = x
    x = x.view(N, NUM_GROUPS , C // NUM_GROUPS * H * W)
    mean = x.mean(-1, keepdim=True)
    var = torch.pow(x - mean, 2).mean(-1, keepdim=True)
    var_inv = 1 / torch.sqrt(var + eps)

    w = w.view(1, C, 1, 1)
    bias = bias.view(1, C, 1, 1)
    x_ = (x - mean) * var_inv
    x_ = x_.view(N,C,H,W)
    y  = w * x_ + bias
    return (y, mean, var_inv)

def simulate_backward(grad_o, mean, var_inv, x, net):
    mean1 = mean.view(N, NUM_GROUPS, 1)      # (N, NUM_GROUG, 1, 1) => (N, NUM_GROUG, 1)
    var1  = var_inv.view(N, NUM_GROUPS, 1)   # (N, NUM_GROUG, 1, 1) => (N, NUM_GROUP, 1)
    grad_o1 = grad_o.view(N, NUM_GROUPS, C//NUM_GROUPS * H * W)

    w = net.weight.view(1, C, 1, 1) # 1, C, 1, 1
    b = net.bias.view(1, C, 1, 1)   # 1, C, 1, 1
    M = H * W * (C // NUM_GROUPS)

    x1 = x.view(N, NUM_GROUPS, C//NUM_GROUPS * H * W)

    x_hat = (x1 - mean1) * var1 # N, Num_GROUPS, C//NUM_GROUPS * H * W

    dw = (grad_o1 * x_hat)
    dw = dw.view(N, C, -1).sum(0).sum(-1, keepdim=False) # C
    db = grad_o.sum(0, keepdim=False).sum(-1, keepdim=False).sum(-1, keepdim=False) # C,

    d_xhat = grad_o * w # [N, C, H, W]
    d_xhat1 = d_xhat.view(N, NUM_GROUPS, -1) # N, Num_GROUPS, C//NUM_GROUPS * H * W

    dx = (d_xhat1 - d_xhat1.mean(-1, keepdim=True) - torch.mean(d_xhat1 * x_hat, dim=-1, keepdim=True) * x_hat) * var1 * 2
    dx = dx.view(N, C, H, W)

    # // InputNormalized = ( Input − SavedMean ) × SavedInvstd
    # // GradWeight = ∑<n, h, w> GradOutput × InputNormalized
    # // GradBias = ∑<n, h, w> GradOutput
    # // GradInput = Weight × SavedInvstd × ( GradOutput − ( InputNormalized × GradWeight + GradBias ) / NHW )
    #              = Weight x SavedInvstd x GradOutput - Weight × SavedInvstd x InputNormalized × GradWeight / NHW                              - Weight × SavedInvstd × GradBias / NHW
    #              = SavedInvstd x (Weight x GradOutput) - (SavedInvstd/NHW) x Weight x InputNormalized × GradWeight                            - (SavedInvstd/NHW) x Weight x GradBias
    #              = SavedInvstd x d_xhat                - (SavedInvstd/NHW) x InputNormalized x Weight x  ∑GradOutput × InputNormalized        - (SavedInvstd/NHW) x Weight x ∑GradOutput
    #              = SavedInvstd x d_xhat                - (SavedInvstd/NHW) x InputNormalized x ∑ Weight x GradOutput × InputNormalized        - (SavedInvstd/NHW) x ∑d_xhat
    #              = SavedInvstd x d_xhat                - (SavedInvstd/NHW) x InputNormalized x ∑ (d_xhat  × InputNormalized)                  - (SavedInvstd/NHW) x ∑d_xhat

    #import pdb;pdb.set_trace()

    return dx, dw, db

if __name__ == "__main__":
    inp = torch.range(1,N*C*H*W).view(N,C,H,W)
    inp = torch.randn(N,C,H,W)
    inp_tpu = inp.to(device) #.half()
    inp.requires_grad = True
    inp_tpu.requires_grad = True

    net = nn.GroupNorm(num_channels=C, num_groups=NUM_GROUPS, eps=eps)
    net_tpu = copy.deepcopy(net).to(device) #.half()
    
    net.requires_grad_ = False
    net_tpu.requires_grad_ = False
    ### forward    
    res_cpu = net(inp)
    torch_tpu.tpu.OpTimer_reset()
    res_tpu = net_tpu(inp_tpu)
    torch_tpu.tpu.OpTimer_dump()
    (res_simulate, mean, var_inv) = simulate_forward(inp, net)

    grad_o = torch.rand(res_cpu.shape)
    grad_o_tpu = grad_o.to(device) #.half()

    ### backward
    t1 = time.time()
    res_cpu.backward(grad_o)
    t2 = time.time()
    print(t2 - t1)
    torch_tpu.tpu.OpTimer_reset()
    res_tpu.backward(grad_o_tpu)
    torch_tpu.tpu.OpTimer_dump()
    dx, dw, db = simulate_backward(grad_o, mean, var_inv, inp, net)

    forward_diff = res_cpu - res_simulate
    inp_grad_diff = inp.grad - inp_tpu.grad.cpu()
    dw_diff = net.weight.grad - net_tpu.weight.grad.cpu()
    db_diff = net.bias.grad - net_tpu.bias.grad.cpu()

    print("out:", torch.max(abs(forward_diff)))
    print("inp_grad:", torch.max(abs(inp_grad_diff)))
    print("dw:", torch.max(abs(dw_diff)))
    print("db:", torch.max(abs(db_diff)))
    import pdb;pdb.set_trace()