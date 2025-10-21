import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch_tpu
torch.manual_seed(1000)
device = "tpu"

def case1():
    a = torch.range(1, 6).to(torch.float).view(2,3)
    a_tpu = a.to(device)
    import pdb; pdb.set_trace()
    torch.tpu.OpTimer_reset()
    a_tpu = a_tpu.transpose(1,0)
    b_tpu = a_tpu.contiguous()
    torch.tpu.OpTimer_dump()

    import pdb; pdb.set_trace()

def case2():
    a = torch.range(1, 6).to(torch.float).view(2,3)
    a_tpu = a.to(device)
    torch.tpu.OpTimer_reset()
    a_tpu = a_tpu.transpose(1,0)
    b_tpu = a_tpu + 1
    torch.tpu.OpTimer_dump()
    import pdb; pdb.set_trace()

def case3():
    a = torch.empty_strided((2, 3), (1, 3), dtype=torch.float32)
    a.requires_grad = True
    a_c = a.contiguous()
    ones  = torch.range(1, 6).to(torch.float).view(2, 3)
    import pdb; pdb.set_trace()
    a_c.backward(ones)
    import pdb; pdb.set_trace()
    torch.tpu.OpTimer_reset()
    a_tpu = torch.empty_strided((2, 3), (1, 3), dtype=torch.float32, device='tpu')
    a_tpu.requires_grad = True
    a_tpu_c = a_tpu.contiguous()
    ones_tpu  = torch.range(1, 6).to(torch.float).view(2, 3).to('tpu')
    torch.tpu.OpTimer_dump()
    
    torch.tpu.OpTimer_reset()
    a_tpu_c.backward(ones_tpu)
    torch.tpu.OpTimer_dump()
    import pdb; pdb.set_trace()


def case_mergeSplitpermute_sequence_256(use_fp16 = False):
    device = "tpu"
    batch = 32
    head_num = 12
    hidden_size = 768
    sequence = 256

    inp = torch.randn(batch, sequence, hidden_size * 3)
    inp_tpu = inp.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    print("=======cpu=====")
    q,k,v = inp.split(hidden_size, -1)
    q1 =q.view(batch, sequence, head_num, hidden_size // head_num)
    q2 = q1.permute(0, 2, 1, 3)
    q3 = q2.transpose(-1,-2)
    q4 = q3.contiguous()
    print(q3.stride())
    print(q4.stride())

    print("=======tpu=====")
    q_tpu,_,__ = inp_tpu.split(hidden_size, -1)
    q1_tpu =q_tpu.view(batch, sequence, head_num, hidden_size // head_num)
    q2_tpu = q1_tpu.permute(0, 2, 1, 3)
    q3_tpu = q2_tpu.transpose(-1,-2)
    q4_tpu = q3_tpu.contiguous()
    print(q3_tpu.stride())
    print(q4_tpu.stride())

    print("=====compare======")
    diff = q4 - q4_tpu.cpu()
    print("diff", torch.max(abs(diff)))

if __name__ == "__main__":
    # case1()
    # case2()
    case3()
    #case_mergeSplitpermute_sequence_256(True)
