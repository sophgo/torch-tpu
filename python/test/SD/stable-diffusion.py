import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from transformers import CLIPTextModel, CLIPTokenizer
import sys 
sys.path.append("..") 
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../../libtorch_plugin/build/liblibtorch_plugin.so"
torch.ops.load_library(PLUGIN_PATH)
optimer = Optimer(PLUGIN_PATH)
device = "privateuseone"
import os
os.environ["CMODEL_GLOBAL_MEM_SIZE"]="12000000000"

pretrained_model_name = "CompVis/stable-diffusion-v1-4"
RANK = 4
N = 1
C = 3
H = 512
W = 512

def all_model():
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name, subfolder="tokenizer", revision=None
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name, subfolder="unet", revision=None
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    #unet.to(device)
    vae.to(device)
    text_encoder.to(device)

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
    lora_layers = AttnProcsLayers(unet.attn_processors)
    unet.train()

    import pdb;pdb.set_trace()
    pixel_input = torch.randn((N,C,H,W)).to(device)
    text_input = torch.randint(0, text_encoder.config.vocab_size, (1, text_encoder.config.max_position_embeddings)).to(device)
    latents = vae.encode(pixel_input).latent_dist.sample()

    noise = torch.randn_like(latents).to(device)

    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (N,), device=latents.device)
    timesteps = timesteps.long()

    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    encoder_hidden_states = text_encoder(text_input)[0]

    pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    target = noise

    loss = F.mse_loss(pred.float(), target.float(), reduce="none")
    print("loss: ", loss)

if __name__ == "__main__":
    all_model()