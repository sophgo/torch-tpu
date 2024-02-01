import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch_tpu
torch.manual_seed(1000)

def case_4d():
    """
    [N C H W] => [N C W H] 
    """
    ############## config ###################
    device = "tpu"
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

def case_3d():
    """
    """
    ############## config ###################
    device = "tpu"
    N = 128
    C = 512
    H = 512
    #########################################
    a = torch.ones(N,C,H)
    a_tpu = a.to(device).half()
    print("before stride", a_tpu.stride())

    a_trans_tpu = torch.transpose(a_tpu, 1, 2)
    print("a_trans_tpu's stride: ", a_trans_tpu.stride())
    print("a_trans_tpu.storage", a_trans_tpu.storage().data_ptr())

    a_t2 = a_trans_tpu.contiguous()
    print("a_t2 stride" ,a_t2.stride())
    print("a_t2.storage", a_t2.storage().data_ptr())

    # a_tpu2 = torch.transpose(a_t2, 1, 2)
    # print("a_tpu2" ,a_tpu2.stride(), a_tpu2.is_contiguous())

    # a_cont_tpu = a_tpu2.contiguous()

def case_2d():
    """
    """
    ############## config ###################
    device = "tpu"
    N = 128
    C = 512
    H = 512
    #########################################
    a = torch.ones(N, C * H)
    a_tpu = a.to(device).half()
    print("before stride", a_tpu.stride())

    a_trans_tpu = torch.transpose(a_tpu, -1, -2)
    print("a_trans_tpu's stride: ", a_trans_tpu.stride())
    print("a_trans_tpu.storage", a_trans_tpu.storage().data_ptr())

    a_t2 = a_trans_tpu.contiguous()
    print("a_t2 stride" ,a_t2.stride())
    print("a_t2.storage", a_t2.storage().data_ptr())



if __name__ == "__main__":
    case_3d()