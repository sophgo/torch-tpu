import copy
import torch
import torch.nn as nn
from utils import ForwardHack, BackwardHack,DumpIns
DI = DumpIns()
# torch.ops.load_library("../../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
device = "privateuseone:0"

# def dump_backward():
#     """
#     embedding
#     """
#     device = "privateuseone"

#  ##########################
#     batch      = 1
#     sequence   = 72
#     vocab_size = 100
#     embed_dim  = 768
#     ##########################
#     batch      = 6
#     sequence   = 4096
#     vocab_size = 2
#     embed_dim  = 12288
#     DI.dump("Copy_input_BS")
#     inp = torch.randint(0, vocab_size, (batch, sequence)).int()
#     inp =  inp.view((batch, sequence))

#     if inp.dtype==torch.int64:
#         assert(torch.max(torch.abs(inp))<65535)

#     inp_tpu = inp.to(device)
#     DI.dump("grad_W_BS_H")
#     ref = torch.ones(batch, sequence, embed_dim)
#     ref_tpu = ref.to(device)
#     DI.dump("Embedding_FP_Index_Select")
#     net = nn.Embedding(vocab_size, embed_dim)
#     DI.dump("Embedding_Weight_V_H")
#     net.weight = nn.Parameter(torch.ones((vocab_size, embed_dim)))
#     net_tpu = copy.deepcopy(net)
#     net_tpu = net_tpu.to(device) #.half()
#     print("============start forward ========")
#     out_tpu = net_tpu(inp_tpu)

#     print("============start backward ========")
#     DI.dump("Embedding_BP")
#     out_tpu.backward(ref_tpu)



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

    batch      = 1
    sequence   = 1
    vocab_size = 2
    embed_dim  = 12

    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.Embedding = nn.Embedding(vocab_size, embed_dim)
                self.Embedding.weight = nn.Parameter(torch.rand((vocab_size, embed_dim)))

            def forward(self, a1):
                # a2 = self.Embedding(a1)
                # a2 = BackwardHack.apply("Embedding_BP_internal", a2)
                a2 = torch.where(a1.bool(), a1.float(), a1.float())
                a2 = BackwardHack.apply("Embedding_BP_internal", a2)
                return a2

    #inp = torch.randint(0, vocab_size, (batch, sequence)).int()
    inp = torch.range(0, batch * sequence - 1)
    inp = torch.randint(0, vocab_size, (batch, sequence)).int()

    if inp.dtype==torch.int64:
        assert(torch.max(torch.abs(inp))<65535)

    inp =  inp.view((batch, sequence))#.int()
    inp_tpu = inp.to(device)

    # ref = torch.ones(batch, sequence, embed_dim).to(torch.int32)
    ref = torch.randint(0, 2, (batch, sequence, embed_dim)).to(torch.int32)
    ref_tpu = ref.to(device) #.half()

    net = Test_Module()
    net_tpu = copy.deepcopy(net)
    net_tpu = net_tpu.to(device) #.half()
    print("============start forward ========")
    out  = net(inp)
    DI.dump("Embedding_Forward")
    out_tpu = net_tpu(inp_tpu)

    print("cpu",out.cpu())
    print("tpu",out_tpu.cpu())
    # print("diff forward tpu-cpu",torch.sum(torch.abs(out.cpu().flatten()-out_tpu.cpu().flatten())))

    # print("============compare result =======")
    # diff = out - out_tpu.cpu()
    # print("max diff : ", torch.max(abs(diff)))

    print("============start backward ========")
    out.backward(ref)
    DI.dump("Embedding_Backward")
    out_tpu.backward(ref_tpu)
    print("cpu",out.cpu())
    print("tpu",out_tpu.cpu())
    print("diff tpu-cpu",torch.sum(torch.abs(out.cpu().flatten()-out_tpu.cpu().flatten())))
    #OPT.dump()
    print("============compare grad =======")
    print("cpu",net.Embedding.weight.grad.cpu())
    print("tpu",net_tpu.Embedding.weight.grad.cpu())
    print("diff grad tpu-cpu",torch.sum(torch.abs(net.Embedding.weight.grad.cpu().flatten()-net_tpu.Embedding.weight.grad.cpu().flatten())))

if __name__ == "__main__":
    case_embedding_backward()
    # dump_backward()