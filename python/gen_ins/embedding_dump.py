import copy
import torch
import torch.nn as nn
from utils import ForwardHack, BackwardHack, DumpIns
DI = DumpIns()


def dump_backward():
    """
    embedding 
    """
    device = "privateuseone"

 ##########################
    batch      = 1
    sequence   = 72
    vocab_size = 100
    embed_dim  = 768
    ##########################
    batch      = 6
    sequence   = 4096
    vocab_size = 2
    embed_dim  = 12288
    DI.dump("Copy_input_BS")
    inp = torch.randint(0, vocab_size, (batch, sequence)).int()
    inp =  inp.view((batch, sequence))

    if inp.dtype==torch.int64:
        assert(torch.max(torch.abs(inp))<65535)

    inp_tpu = inp.to(device)
    DI.dump("grad_W_BS_H")
    ref = torch.ones(batch, sequence, embed_dim)
    ref_tpu = ref.to(device) 
    DI.dump("Embedding_FP_Index_Select")
    net = nn.Embedding(vocab_size, embed_dim)
    DI.dump("Embedding_Weight_V_H")
    net.weight = nn.Parameter(torch.ones((vocab_size, embed_dim)))
    net_tpu = copy.deepcopy(net)
    net_tpu = net_tpu.to(device) #.half()
    print("============start forward ========")
    out_tpu = net_tpu(inp_tpu)

    print("============start backward ========")
    DI.dump("Embedding_BP")
    out_tpu.backward(ref_tpu)

if __name__ == "__main__":
    dump_backward()