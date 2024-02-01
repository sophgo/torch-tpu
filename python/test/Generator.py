import torch
import torch_tpu
device = torch.device("tpu:0")
g_tpu = torch.manual_seed(32)

latents = torch.rand((2,3,4), generator=g_tpu).to(device)
import pdb;pdb.set_trace()