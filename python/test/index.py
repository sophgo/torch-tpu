import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    # the num of indexs == 1
    cases = [ 
                #    [(3, 804, 7), (3, 804), torch.bool ],
                #    [(5, 995, 7), (5, 995), torch.bool ],
                #    [(5, 995, 2), (5, 995), torch.bool ],
                #    [(5, 1326, 7), (5, 1326), torch.bool ],
                #    [(5, 1326, 2), (5, 1326), torch.bool ],
                #    [(5, 1081, 7), (5, 1081), torch.bool ],
                #    [(5, 1081, 2), (5, 1081), torch.bool ],
                #    [(7), (2,), torch.int32 ],
                   [(3,2), (2974,), torch.int32 ],
                   [(3,2), (3969,), torch.int32 ],
                  ]
    for (ishape, indshape, inddype) in cases:
        inp = torch.randn(ishape)
        inp_tpu = inp.to(device)
        if inddype == torch.bool:
            ind = torch.randint(0, 2, indshape).bool()
        elif inddype == torch.int32:
            ind = torch.randint(0, 2, indshape).to(inddype)
        ind_tpu = ind.to(device)
        o_cpu = inp[ind]
        o_tpu = inp_tpu[ind_tpu]
        diff = o_cpu - o_tpu.cpu()
        print("input : ", inp)
        print("index : ", ind)
        print("cpu : ", o_cpu)
        print("tpu : ", o_tpu.cpu())
        print(f"max diff : {torch.max(abs(diff))}")
        import pdb;pdb.set_trace()
    
def case2():
    cases = [   # inp,               ind0,     ind1,    ind2,    ind3,  ind-dtype
                [(64, 3, 80, 80, 85), (2974,), (2974,), (2974,), (2974,), torch.int32],
                [(64, 3, 40, 40, 85), (3969,), (3969,), (3969,), (3969,), torch.int32],
                [(64, 3, 20, 20, 85), (3238,), (3238,), (3238,), (3238,), torch.int32],
            ]
    for (ishape, ind0shape, ind1shape, ind2shape, ind3shape, inddype) in cases:
        inp = torch.randn(ishape)
        inp_tpu = inp.to(device)
        ind0 = torch.randint(0, inp.shape[0], ind0shape).to(inddype)
        ind0_tpu = ind0.to(device)     
        ind1 = torch.randint(0, inp.shape[1], ind1shape).to(inddype)
        ind1_tpu = ind1.to(device)     
        ind2 = torch.randint(0, inp.shape[2], ind2shape).to(inddype)
        ind2_tpu = ind2.to(device)     
        ind3 = torch.randint(0, inp.shape[3], ind3shape).to(inddype)
        ind3_tpu = ind3.to(device)
        o_cpu = inp[ ind0, ind1, ind2, ind3]
        o_tpu = inp_tpu[ ind0_tpu, ind1_tpu, ind2_tpu, ind3_ tpu]
        
        diff = o_cpu - o_tpu.cpu()
        print("input : ", inp)
        print("cpu : ", o_cpu)
        print("tpu : ", o_tpu.cpu())
        print(f"max diff : {torch.max(abs(diff))}")
        import pdb;pdb.set_trace()

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

def case6():
    boxes = torch.randn((3, 5))
    boxes_tpu = boxes.to(device)

    boxes[..., [0, 2]] -= 16.0
    boxes_tpu[..., [0,2]] -= 16.0

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    # case1()
    case2()
    # case3()
    # case4()
    # case5()
    # case6()