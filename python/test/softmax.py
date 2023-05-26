import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

if __name__ == "__main__":
    device = "privateuseone"
    batch = 32
    head_num = 12
    sequence = 256
    hidden_size = 768
    inp_cpu = torch.rand(batch, head_num, sequence, sequence)
    inp_tpu = inp_cpu.to(device)
    t1 = time.time()
    inp_tpu = inp_tpu.half()
    t2 = time.time()
    print("dtype convert", t2 - t1)

    t1 = time.time()
    inp_tpu = inp_tpu.float()
    t2 = time.time()
    print("dtype convert", t2 - t1)
    import pdb; pdb.set_trace()

    net_cpu = nn.Softmax(-1)
    net_tpu = copy.deepcopy(net_cpu)
    net_tpu = net_tpu.to(device) #.half()

    t1 = time.time()
    out_cpu = net_cpu(inp_cpu)
    t2 = time.time()
    print("cpu time ",t2 -t1)

    t1 = time.time()
    out_tpu = net_tpu(inp_tpu)
    t2 = time.time()
    print("tpu time ",t2 -t1)

    out_tpu = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu
    # print("cpu_out")
    # print(out_cpu)
    # print("tpu_out")
    # print(out_tpu)
    
    print (torch.max(abs(out_diff)))
