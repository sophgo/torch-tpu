import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    B = 10
    S = 34
    H = 34
    inp = torch.randn((B,S,H))
    B_ind = torch.LongTensor([1, 2])
    S_ind = torch.randint(0, S, [1,34,4])
    inp_tpu = copy.deepcopy(inp).to(device)
    B_ind_tpu = copy.deepcopy(B_ind).to(device)
    S_ind_tpu = copy.deepcopy(S_ind).to(device)
    
    o_cpu = inp[B_ind]
    o_tpu = inp_tpu[B_ind_tpu]

    diff = o_cpu - o_tpu.cpu()
    print("input : ", inp)
    print("cpu : ", o_cpu)
    print("tpu : ", o_tpu.cpu())
    print(f"max diff : {torch.max(abs(diff))}")
    import pdb;pdb.set_trace()
    
def case2():
    a = torch.arange(100).reshape(10, 10).float()
    index0 = torch.arange(0, 10).int()
    index1 = torch.randint(0, 10, (10, )).int()
    index2 = torch.randint(0, 2, (10,)).bool()

    a_tpu = a.to(device)
    index0_tpu = index0.to(device)
    index1_tpu = index1.to(device)
    index2_tpu = index2.to(device)
    
    a[index0, index1] += 127.
    a_tpu[index0_tpu, index1_tpu] += 127.
    
    a[index2] = 255
    a_tpu[index2_tpu] = 255

    print('cpu')
    print(index0, index1)
    print(a)
    print('tpu')
    print(index0_tpu.cpu(), index1_tpu.cpu())
    print(a_tpu.cpu())
    print('diff', torch.max(torch.abs(a - a_tpu.cpu())))

def case3():
    w = torch.randn((4001, 8192), dtype=torch.float16)
    w_tpu = w.to(device)

    max_diff = {}
    num = 1
    while (num <= 16*4096):
        index = torch.randint(0, 4001, (num, ), dtype=torch.int32)

        index_tpu = index.to(device)
        tpu_out = w_tpu[index_tpu]
        cpu_out = w[index]

        print(tpu_out.shape, cpu_out.shape)
        # print(f'num: {num}, max diff: {torch.max(torch.abs(cpu_out - tpu_out.cpu()))}')
        max_diff[num] = torch.max(torch.abs(cpu_out - tpu_out.cpu()))
        num *= 2
    
    print(max_diff)

def case4():
    '''int32'''
    t = torch.range(1, 2 * 3 * 4).view((2,3, 4))
    t_tpu = t.to(device)
    
    j = torch.randint(0, 2, (3, 4),dtype=torch.int32)
    j_tpu = j.to(device)

    to = t[j]
    to_tpu = t_tpu[j_tpu]
    print(to_tpu.shape)
    print(to.shape)

    diff = torch.max(abs(to - to_tpu.cpu()))
    print(diff)

    import pdb; pdb.set_trace()

def case5():
    '''int32'''
    t = torch.range(1, 2 * 3 * 4).view((2,3, 4))
    t_tpu = t.to(device)
    
    j = torch.randint(0, 2, (2, 3),dtype=torch.bool)
    j_tpu = j.to(device)

    to = t[j]
    to_tpu = t_tpu[j_tpu]
    print(to_tpu.shape)
    print(to.shape)

    diff = torch.max(abs(to - to_tpu.cpu()))
    print(diff)

    import pdb; pdb.set_trace()



if __name__ == "__main__":
    # case1()
    # case2()
    # case3()
    # case4()
    case5()