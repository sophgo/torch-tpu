import torch
from torch.optim import Adam

from functools import partial

def fuse_torch_adam():
    torch.optim.Adam = partial(Adam, fused=True)