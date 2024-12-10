import torch
import torch_tpu

device = "tpu:0"

a = torch.randn(3, 3).to(device)
b = torch.randn(3, 3).to(device)
c = a + b
print(c.cpu())