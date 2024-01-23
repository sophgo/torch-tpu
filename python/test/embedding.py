import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import compare_model_grad, Optimer

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"
OPT = Optimer()

def case_embedding():
    ##########################
    batch      = 32
    sequence   = 256
    vocab_size = 50257
    embed_dim  = 768
    ##########################
    batch      = 6
    sequence   = 4096
    vocab_size = 2
    embed_dim  = 12288
    inp = torch.randint(0, vocab_size, (batch, sequence)).int()
    inp_tpu = inp.to(device)
    if inp.dtype==torch.int64:
        assert(torch.max(torch.abs(inp))<65535)

    net = nn.Embedding(vocab_size, embed_dim)
    net_tpu = copy.deepcopy(net)
    net_tpu = net_tpu.to(device)
    print("============start forward ========")
    out  = net(inp)
    out_tpu = net_tpu(inp_tpu)

    print("============compare result =======")
    diff = out - out_tpu.cpu()
    print("max diff : ", torch.max(abs(diff)))
    print("max of inp abs",torch.max(torch.abs(inp)))
    print("diff tpu-cpu",torch.sum(torch.abs(out.cpu().flatten()-out_tpu.cpu().flatten())))



def case_embedding_backward():
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

    #inp = torch.randint(0, vocab_size, (batch, sequence)).int()
    inp = torch.range(0, batch * sequence - 1)
    inp = torch.randint(0, vocab_size, (batch, sequence)).int()

    if inp.dtype==torch.int64:
        assert(torch.max(torch.abs(inp))<65535)

    inp =  inp.view((batch, sequence))#.int()
    print(inp)
    inp_tpu = inp.to(device)

    # ref = torch.ones(batch, sequence, embed_dim).to(torch.int32)
    ref = torch.randint(0, 2, (batch, sequence, embed_dim)).to(torch.int32)
    ref_tpu = ref.to(device) #.half()

    net = nn.Embedding(vocab_size, embed_dim)
    net.weight = nn.Parameter(torch.ones((vocab_size, embed_dim)))
    net_tpu = copy.deepcopy(net)
    net_tpu = net_tpu.to(device) #.half()
    print("============start forward ========")
    out  = net(inp)
    #OPT.reset()
    out_tpu = net_tpu(inp_tpu)
    #OPT.dump()
    print("cpu",out.cpu())
    print("tpu",out_tpu.cpu())
    # print("diff forward tpu-cpu",torch.sum(torch.abs(out.cpu().flatten()-out_tpu.cpu().flatten())))

    # print("============compare result =======")
    # diff = out - out_tpu.cpu()
    # print("max diff : ", torch.max(abs(diff)))

    print("============start backward ========")
    out.backward(ref)
    #OPT.reset()
    out_tpu.backward(ref_tpu)

    print("cpu",out.cpu())
    print("tpu",out_tpu.cpu())
    print("diff tpu-cpu",torch.sum(torch.abs(out.cpu().flatten()-out_tpu.cpu().flatten())))
    #OPT.dump()
    print("============compare grad =======")
    print("cpu",net.weight.grad.cpu())
    print("tpu",net_tpu.weight.grad.cpu())
    print("diff grad tpu-cpu",torch.sum(torch.abs(net.weight.grad.cpu().flatten()-net_tpu.weight.grad.cpu().flatten())))
    #import pdb;pdb.set_trace()
    print("inp", inp)
    print('ref',ref)
    compare_model_grad(net, net_tpu)

def case_embedding_backward_simulate():
    """
    test case for simulate embedding's behavior
    """
    ##########################
    batch      = 2
    sequence   = 3
    vocab_size = 8
    embed_dim  = 4
    ##########################

    inp = torch.randint(0, vocab_size, (batch, sequence))
    ref = torch.randn(batch,  sequence, embed_dim)

    net = nn.Embedding(vocab_size, embed_dim, scale_grad_by_freq=True)
    out  = net(inp)

    out.backward(ref)
    print("inp")
    print(inp)
    print("out")
    print(out)

    print("model.weight:")
    print(net.weight)
    print("model.weight.grad")
    print(net.weight.grad)
    print("grad_in")
    print(ref)

if __name__ == "__main__":
    # case_embedding()
    case_embedding_backward()
    #case_embedding_backward_simulate()