import os
import sys
from turtle import forward 
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import AutoencoderKL

sys.path.append("..") 
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../../build/torch_tpu/libtorch_tpu.so"
torch.ops.load_library(PLUGIN_PATH)
optimer = Optimer(PLUGIN_PATH)
device = "tpu"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="12000000000"

pretrained_model_name = "CompVis/stable-diffusion-v1-4"
RANK = 4
N = 1
C = 3
H = 512
W = 512

def from_here():
    pass

def from_diffusers():
    device = "cpu"
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae", revision=None)
    vae.requires_grad_(False)
    vae.to(device)

    pixel_input = torch.randn((N,C,H,W)).to(device)
    
    latents = vae.encode(pixel_input).latent_dist.sample()
    import pdb;pdb.set_trace()
if __name__ == "__main__":
    from_diffusers()