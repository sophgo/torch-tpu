import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")

def case1(use_fp16):
    ############## config ###################
    device = "privateuseone"
    batch = 2
    sequence = 8
    hidden_size = 768
    #########################################

    dq = torch.rand(batch, sequence, hidden_size)
    dk = torch.rand(batch, sequence, hidden_size)
    dv = torch.rand(batch, sequence, hidden_size)

    dq_tpu = dq.to(device)
    dk_tpu = dk.to(device)
    dv_tpu = dv.to(device)
    if use_fp16:     
        dq_tpu = dq_tpu.half()
        dk_tpu = dk_tpu.half()
        dv_tpu = dv_tpu.half()

    print("=======forward")
    out = torch.concatenate((dq,dk,dv), dim=-1)
    out_tpu = torch.concatenate((dq_tpu, dk_tpu, dv_tpu), dim=-1).half()
    if use_fp16: 
        out_tpu = out_tpu.half()

    print("============compare result =======")
    diff = out - out_tpu.cpu()
    print("max diff : ", torch.max(abs(diff)))
    print("max diff : ", out_tpu.cpu())
def case2():
    """split backward
    """
    ############## config ###################
    device = "privateuseone"
    batch = 2
    sequence = 8
    hidden_size = 768
    #########################################
    inp = torch.rand(batch, sequence, hidden_size * 3)
    inp_tpu = inp.to(device)

    out_grad = torch.ones(batch, sequence, hidden_size)
    out_grad_tpu = out_grad.to(device)

    # 2.split
    q,k,v = inp.split(hidden_size, dim=2)
    q_tpu,k_tpu,v_tpu = inp_tpu.split(hidden_size, dim=2)

    out = q + k + v
    out_tpu = q_tpu + k_tpu + v_tpu
    
    out.backward(out_grad)
    out_tpu.backward(out_grad_tpu)
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    case1(use_fp16 =1 )
    # case2()