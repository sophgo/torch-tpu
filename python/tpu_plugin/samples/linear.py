import torch
import torch.nn as nn
import copy

from tpu_plugin import TPU
device = TPU(0)

def case_forward(use_fp16=False):
    batch = 8
    sequence = 1024
    hidden_size = 768
    out_size = 3

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp_cpu.to(device).half()

    ln_cpu = nn.Linear(hidden_size, out_size)
    ln_tpu = copy.deepcopy(ln_cpu)
    ln_tpu = ln_tpu.to(device).half()

    out_cpu = ln_cpu(inp_cpu)
    out_tpu = ln_tpu(inp_tpu)
    out_tpu = out_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu
    print(out_diff)

if __name__ == "__main__":
    case_forward()