import torch
import torch_tpu


def hello_world(device):
    a = torch.randn(3, 3).to(device)
    b = torch.randn(3, 3).to(device)
    c = a + b
    print(c.cpu())


hello_world("tpu:0")
print(torch.tpu.device_count())
