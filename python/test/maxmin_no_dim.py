import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def test_max():
    a_origin = torch.tensor(3).to(torch.float32)
    a = a_origin.to(device)
    max_values  = torch.max(a_origin)
    max_values_tpu = torch.max(a)

    output_cpu = max_values[0] if isinstance(max_values, tuple) else max_values
    output_tpu = max_values_tpu.to("cpu")
    print("output_tpu : ", output_tpu)
    print("delta : ", torch.equal(output_cpu, output_tpu))
    print(output_cpu.item() == output_tpu.item())

    a_origin=torch.rand(4, 6, 29).to(torch.float32)
    a = a_origin.to(device)
    max_values  = torch.max(a_origin)
    max_values_tpu = torch.max(a)

    output_cpu = max_values[0] if isinstance(max_values, tuple) else max_values
    output_tpu = max_values_tpu.to("cpu")
    print("output_tpu : ", output_tpu)
    print("delta : ", torch.equal(output_cpu, output_tpu))
    print(output_cpu.item() == output_tpu.item())

def test_min():
    a_origin = torch.tensor(3).to(torch.float32)
    a = a_origin.to(device)
    min_values  = torch.min(a_origin)
    min_values_tpu = torch.min(a)

    output_cpu = min_values[0] if isinstance(min_values, tuple) else min_values
    output_tpu = min_values_tpu.to("cpu")
    print("output_tpu : ", output_tpu)
    print("delta : ", torch.equal(output_cpu, output_tpu))
    print(output_cpu.item() == output_tpu.item())

    a_origin=torch.rand(4, 6, 29).to(torch.float32)
    a = a_origin.to(device)
    min_values  = torch.min(a_origin)
    min_values_tpu = torch.min(a)

    output_cpu = min_values[0] if isinstance(min_values, tuple) else min_values
    output_tpu = min_values_tpu.to("cpu")
    print("output_tpu : ", output_tpu)
    print("delta : ", torch.equal(output_cpu, output_tpu))
    print(output_cpu.item() == output_tpu.item())

if __name__ == "__main__":
    test_max()
    test_min()
