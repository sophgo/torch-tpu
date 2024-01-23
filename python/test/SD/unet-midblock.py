import os
import sys
from turtle import forward 
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.models.unet_2d_blocks import UNetMidBlock2DCrossAttn

sys.path.append("..") 
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../../build/torch_tpu/libtorch_tpu.so"
torch.ops.load_library(PLUGIN_PATH)
OPT = Optimer(PLUGIN_PATH)
device = "tpu"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="12000000000"

def test_MidBlock():
    latent_img_size = (1, 1280, 32, 32)
    temb_size = (1, 1280)
    text_emb_size = (1, 77, 2048)

    latent_img = torch.rand(latent_img_size).to(device)
    temb  = torch.rand(temb_size).to(device)
    text_emb = torch.rand(text_emb_size).to(device)
    net = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=10,
                in_channels=1280,
                temb_channels=1280,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                cross_attention_dim=2048,
                num_attention_heads=20,
                resnet_groups=32,
                dual_cross_attention=False,
                use_linear_projection=True,
                upcast_attention=None,
                attention_type='default'
            ).to(device)
    OPT.reset()
    out = net( latent_img,
               temb,
               encoder_hidden_states= text_emb,
               attention_mask=None,
               cross_attention_kwargs=None,
               encoder_attention_mask=None,
            )
    OPT.dump()

    

if __name__ == "__main__":
    test_MidBlock()
