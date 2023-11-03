import os
import sys
from turtle import forward 
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor

sys.path.append("..") 
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../../build/torch_tpu/libtorch_tpu.so"
torch.ops.load_library(PLUGIN_PATH)
optimer = Optimer(PLUGIN_PATH)
device = "privateuseone"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="12000000000"

pretrained_model_name = "CompVis/stable-diffusion-v1-4"
N = 1
C = 4
H = 64
W = 64
B = N
S = 77
H = 768
num_train_timesteps = 2000

RANK = 4
def from_here():
    pass

def from_diffusers():
    #device = "cpu"
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name, subfolder="unet", revision=None
    )
    unet.requires_grad_(False)
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=RANK,
        )

    unet.set_attn_processor(lora_attn_procs)
    unet.to(device)
    unet.train()

    noisy_latents = torch.randn((N,C,H,W)).to(device)
    timesteps = torch.randint(0, num_train_timesteps, (N,)).int().to(device)
    encoder_hidden_states = torch.randn((B, S, H)).to(device)
    pred = unet(noisy_latents, timesteps, encoder_hidden_states)


if __name__ == "__main__":
    from_diffusers()