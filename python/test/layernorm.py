import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

if __name__ == "__main__":
    device = "privateuseone"
    batch = 8
    sequence = 1024
    hidden_size = 768
    layer_norm_epsilon = 1e-5

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp_cpu.to(device).half()

    ln_cpu = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
    ln_tpu = copy.deepcopy(ln_cpu)
    ln_tpu = ln_tpu.to(device).half()

    out_cpu = ln_cpu(inp_cpu)
    out_tpu = ln_tpu(inp_tpu)
    out_tpu = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu
    print("cpu_out")
    print(out_cpu)
    print("tpu_out")
    print(out_tpu)
    
    print (torch.max(abs(out_diff)))
