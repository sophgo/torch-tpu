import torch
import torch_tpu
device = "privateuseone:1"
g_cpu = torch.Generator()
g_tpu = torch.Generator(device=device)
import pdb;pdb.set_trace()