import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
device = "privateuseone"

def case_embedding():
    ##########################
    batch      = 32
    sequence   = 256
    vocab_size = 50257
    embed_dim  = 768
    ##########################

    inp = torch.randint(0, vocab_size, (batch, sequence)).int()
    inp_tpu = inp.to(device)

    net = nn.Embedding(vocab_size, embed_dim)
    net_tpu = copy.deepcopy(net)
    net_tpu = net_tpu.to(device)
    print("============start forward ========")
    out  = net(inp)
    out_tpu = net_tpu(inp_tpu)

    print("============compare result =======")
    diff = out - out_tpu.cpu()
    print("max diff : ", torch.max(abs(diff)))


if __name__ == "__main__":
    case_embedding()