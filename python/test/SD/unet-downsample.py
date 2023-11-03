import os
import sys
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers.models.unet_2d_blocks import DownBlock2D, CrossAttnDownBlock2D

sys.path.append("..") 
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../../build/torch_tpu/libtorch_tpu.so"
torch.ops.load_library(PLUGIN_PATH)
OPT = Optimer(PLUGIN_PATH)
device = "privateuseone"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="12000000000"

def test_DownBlock2D():
    latent_img_size = (1, 320, 128, 128)
    temb_size = (1, 1280)
    text_emb_size = (1, 77, 2048)

    latent_img = torch.rand(latent_img_size).to(device)
    temb  = torch.rand(temb_size).to(device)
    text_emb = torch.rand(text_emb_size).to(device)

    net = DownBlock2D(
        num_layers = 2,
        in_channels = 320,
        out_channels = 320,
        temb_channels = 1280,
        add_downsample = True,
        resnet_eps = 1e-5,
        resnet_act_fn = "silu",
        resnet_groups = 32,
        downsample_padding = 1,
        resnet_time_scale_shift = 'default'
    ).to(device)

    OPT.reset()
    out = net(hidden_states=latent_img, temb=temb)
    OPT.dump()

def test_CrossAttnDownBlock2D():
    latent_img_size = (1, 320, 64, 64)
    temb_size = (1, 1280)
    text_emb_size = (1, 77, 2048)

    latent_img = torch.rand(latent_img_size).to(device)
    temb  = torch.rand(temb_size).to(device)
    text_emb = torch.rand(text_emb_size).to(device)

    net = CrossAttnDownBlock2D(
            num_layers= 2,
            transformer_layers_per_block= 2,
            in_channels= 320,
            out_channels= 640,
            temb_channels= 1280,
            add_downsample= True,
            resnet_eps= 1e-5,
            resnet_act_fn= "silu",
            resnet_groups= 32,
            downsample_padding= 1,
            cross_attention_dim= 2048,
            num_attention_heads= 10,
            dual_cross_attention= False,
            use_linear_projection= True,
            only_cross_attention= False,
            upcast_attention= None,
            resnet_time_scale_shift= "default",
            attention_type="default"
        ).to(device)

    OPT.reset()
    out = net(hidden_states=latent_img, temb=temb, encoder_hidden_states=text_emb,
              attention_mask=None, cross_attention_kwargs=None, encoder_attention_mask=None)
    OPT.dump()


def test_CrossAttnDownBlock2D_2():
    net = CrossAttnDownBlock2D(
            num_layers= 2,
            transformer_layers_per_block= 10,
            in_channels= 640,
            out_channels= 1280,
            temb_channels= 1280,
            add_downsample= True,
            resnet_eps= 1e-5,
            resnet_act_fn= "silu",
            resnet_groups= 32,
            downsample_padding= 1,
            cross_attention_dim= 2048,
            num_attention_heads= 20,
            dual_cross_attention= False,
            use_linear_projection= True,
            only_cross_attention= False,
            upcast_attention= None,
            resnet_time_scale_shift= "default",
            attention_type="default"
        )

if __name__ == "__main__":
    test_DownBlock2D()
    #test_CrossAttnDownBlock2D()