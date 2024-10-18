import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusers import DDPMScheduler

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
C = 4
H = 64
W = 64
V = 49408

def from_diffusers():
    #device = "cpu"
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

    latents = torch.randn(B, C, H, W).to(device)
    noise   = torch.rand_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,)).to(device)

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    print("noisy_latents : ", noisy_latents.shape)

if __name__ == "__main__":
    from_diffusers()