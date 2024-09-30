import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import compare_model_grad, Optimer

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"
# OPT = Optimer()

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

def case_embedding_backward_large_V():
    device = "tpu"
    batch = 1

    BS = 128      #BS
    V = 152064    #V
    H = 3584      #H
    assert V   < 2**32 - 1
    assert H   < 2**32 - 1
    assert BS  < 2**32 - 1

    target_type =torch.half#torch.half


    model = nn.Embedding(V, H, dtype=target_type)
    X       = torch.randint(0, V, (batch, BS)).to(torch.int32)
    grad_Y  = torch.randn(batch, BS, H, dtype=target_type)
    #model.weight = nn.Parameter(torch.ones([V,  H]))

    W = model.weight
    def embedding_fp(X_input, W):
        X  = X_input.flatten()
        Y = torch.zeros([BS, H],dtype=target_type)
        '''
        Y[i,h] = W[X[i],h]
        '''
        # print("shape_dq",Y.shape,W.shape)
        for i in range(BS):
            # print("shape_dq",Y[[i], :] .shape,W[X[i],:].shape)
            Y[[i], :] = W[X[i],:]
        return Y

    def embedding_bp_ref(grad_Y_, X_input, V):
        X  = X_input.flatten()
        grad_W =  torch.zeros([V, H], dtype=target_type)
        grad_Y = grad_Y_.reshape(BS, H)
        for v in range(V):
            for idx, x in enumerate(X):
                if x == v:
                    grad_W[v, :] += grad_Y[idx, :]
        return grad_W


    if target_type ==  torch.half:
        model_tpu = copy.deepcopy(model).half().to(device)
        X_tpu = copy.deepcopy(X).to(device)
        grad_Y_tpu = copy.deepcopy(grad_Y).half().to(device)
    else:
        model_tpu = copy.deepcopy(model).to(device)
        X_tpu = copy.deepcopy(X).to(device)
        grad_Y_tpu = copy.deepcopy(grad_Y).to(device)
    Y = model(X)
    Y_tpu = model_tpu(X_tpu)

    Y_dq   =  embedding_fp(X, W)
    assert torch.sum(torch.abs(Y_dq -Y)) == 0
    assert torch.sum(torch.abs(Y_dq -Y_tpu.cpu())) == 0
    assert torch.sum(torch.abs(Y -Y_tpu.cpu())) == 0


    Y.backward(grad_Y)
    Y_tpu.backward(grad_Y_tpu)


    grad_W = model.weight.grad
    grad_W_tpu = model_tpu.weight.grad.cpu()

    grad_W_dq = embedding_bp_ref(grad_Y, X, V)

    assert torch.sum(torch.abs(grad_W_dq -grad_W.cpu())) == 0, torch.sum(torch.abs(grad_W_dq -grad_W.cpu()))
    assert torch.sum(torch.abs(grad_W_dq -grad_W_tpu.cpu())) == 0, torch.sum(torch.abs(grad_W_dq -grad_W_tpu.cpu()))
    assert torch.sum(torch.abs(grad_W_tpu -grad_W.cpu())) == 0, torch.sum(torch.abs(grad_W_tpu -grad_W.cpu()))

    diff_output = torch.max(torch.abs(Y - Y_tpu.cpu()))
    print(f"{diff_output = }")

    print("grad_W sumabs",torch.sum(torch.abs(model.weight.grad - model_tpu.weight.grad.cpu())))
    # diff_grads = torch.max(torch.abs(model.weight.grad - model_tpu.weight.grad.cpu()))

    # print(f"{diff_output = }")
    # print(f"{diff_grads = }")



if __name__ == "__main__":
    # case_embedding()
    # case_embedding_backward()
    #case_embedding_backward_simulate()
    case_embedding_backward_large_V()