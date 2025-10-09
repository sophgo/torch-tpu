import torch
from torch.optim import Adam, AdamW

from functools import partial

def fuse_torch_adam():
    torch.optim.Adam = partial(Adam, fused=True)

def fuse_torch_adamw():
    torch.optim.AdamW = partial(AdamW, fused=True)