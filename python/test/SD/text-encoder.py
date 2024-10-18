import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTextModel

sys.path.append("..") 
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../../build/torch_tpu/libtorch_tpu.so"
torch.ops.load_library(PLUGIN_PATH)
optimer = Optimer(PLUGIN_PATH)
device = "tpu"
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="12000000000"
torch.manual_seed(1000)

pretrained_model_name = "CompVis/stable-diffusion-v1-4"
B = 1
S = 77

V = 49408

def from_diffusers():
    #device = "cpu"
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name, subfolder="text_encoder", revision=None
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    #TODO DBUG when dtype == torch.int64
    text_input = torch.randint(0, V, (B, S),dtype=torch.int32).to(device)
    encoder_hidden_states = text_encoder(text_input)[0]


if __name__ == "__main__":
    from_diffusers()