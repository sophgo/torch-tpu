import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

def case_4d():
    """
    [N C H W] => [N C W H] 
    """
    ############## config ###################
    device = "privateuseone"
    batch = 32
    sequence = 256
    hidden_size = 768
    num_heads =12
    attn_head_size = hidden_size // num_heads
    #########################################
    
    inp = torch.randn(batch, num_heads, sequence, attn_head_size)
    inp_tpu = inp.to(device)
    print("inp cpu's stride :", inp.stride())
    print("inp tpu's stride :", inp_tpu.stride())

    print("=========transpose=====")
    o_tran = inp.transpose(-1,-2)
    o_tran_tpu = inp_tpu.transpose(-1,-2)
    print("transed cpu's stride :", o_tran.stride())
    print("transed tpu's stride :", o_tran_tpu.stride())
    
    print("=========compare=====")
    diff = o_tran - o_tran_tpu.cpu()
    # print("=====cpu")
    # print(o_tran)
    # print("=====tpu")
    # print(o_tran_tpu.cpu())
    print("max diff : ", torch.max(abs(diff)))


if __name__ == "__main__":
    case_4d()