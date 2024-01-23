import os
import sys
from turtle import forward 
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.models.unet_2d_blocks import UpBlock2D, CrossAttnUpBlock2D

sys.path.append("..") 
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../../build/torch_tpu/libtorch_tpu.so"
torch.ops.load_library(PLUGIN_PATH)
OPT = Optimer(PLUGIN_PATH)
device = "tpu"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="12000000000"

def test_UpBlock2D():
    UpBlock2D(
            num_layers= 3,
            in_channels=320,
            out_channels=320,
            prev_output_channel=640,
            temb_channels=1280,
            add_upsample=False,
            resnet_eps=1e-5,
            resnet_act_fn='silu',
            resnet_groups=32,
            resnet_time_scale_shift='default',
        )

def test_CrossAttnUpBlock2D():
    CrossAttnUpBlock2D(
            num_layers= 3,
            transformer_layers_per_block= 10,
            in_channels= 640,
            out_channels= 1280,
            prev_output_channel= 1280,
            temb_channels= 1280,
            add_upsample= True,
            resnet_eps=1e-05,
            resnet_act_fn='silu',
            resnet_groups=32,
            cross_attention_dim=2048,
            num_attention_heads=20,
            dual_cross_attention=False,
            use_linear_projection=True,
            only_cross_attention=False,
            upcast_attention=None,
            resnet_time_scale_shift='default',
            attention_type='default'
        )