import torch
import os
import copy
import argparse
from tpu_mlir_jit import aot_backend
import tpu_mlir_jit as tpu_mlir_jit
import pdb
from utils.misc import *
import torch.nn as nn
import torch_tpu
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1684x", choices=['bm1684x', 'bm1690'],
                        help="chip name")
    parser.add_argument("--debug", default="",
                        help="debug")
    parser.add_argument("--cmp", action='store_true',
                        help="enable cmp")
    parser.add_argument("--skip_tpu_mlir", action='store_true',
                        help="skip_tpu_mlir")
    parser.add_argument("--model", default="",help="model name")
    parser.add_argument("--only_test_bwd", action='store_true',help="only_test_bwd")
    parser.add_argument("--fp", default="",help="fp")
    parser.add_argument("--rank",type=int,default=4,help="The dimension of the LoRA update matrices.")

    # from model_transforms
    # parser.add_argument("--model_name", required=True, help="model name")
    # parser.add_argument("--model_def", default = "", help="model definition file.")
    # parser.add_argument("--input_shapes", type=str2shape, default=list(),
    #                     help="list of input shapes, like:[[1,3,224,224],[10],[16]]")
    # parser.add_argument("--input_types", type=str2list, default=list(),
    #                     help="list of input types, like:float32,int32. if not set, float32 as default")
    # parser.add_argument("--output_names", type=str2list, default=list(),
    #                     help="if set, will find names in model and set as real outputs")
    # parser.add_argument("--test_input", default="", type=str2list,
    #                     help="input jpg/npy/npz file for inference, "
    #                     "if has more than one input, join jpg or npy with semicolon")
    # parser.add_argument("--test_result", default="", type=str,
    #                     help="if input is set, result is mlir inference result")
    # parser.add_argument("--cache_skip", action='store_true', help='skip checking the correctness when generate same mlir and bmodel.')
    # parser.add_argument("--tolerance", default='0.99,0.99',
    #                     help="minimum similarity tolerance to model transform")
    # parser.add_argument("--excepts", default='-', help="excepts")
    # parser.add_argument("--add_postprocess", default="", type=str.lower,
    #                     choices=['','yolov3','yolov5','yolov8','ssd','bnr'], help="add postprocess for model")
    # parser.add_argument("--onnx_sim", default="", type=str, choices=['', 'skip_fuse_bn'],
    #                     help="pass options of onnx-sim, sep by quote without space")
    # parser.add_argument("--dump_final_opt", default=True, help='save final_opt onnx file')
    # parser.add_argument("--mlir", type=str, default = "", help="output mlir model file")
    # parser.add_argument("--img", action='store_true', help="generate fake image for CV tasks")
    # parser.add_argument("--output", type=str, default='input.npz', help="output npz/img file")
    # parser.add_argument("--ranges",
    #                     type=str2shape,
    #                     default=list(),
    #                     help="list of input ranges, like:[[0,1],[-10,10]]")
    # # regression test only, not for users
    # parser.add_argument("--patterns_count", type=str2dict, default=dict(),
    #                     help='used for regression test, check if patterns are successfully applied a specific number of times')
    # parser.add_argument("--dynamic_shape_input_names", type=str2list, default=list(),
    #                     help="name list of inputs with dynamic shape, like:input1,input2")
    # parser.add_argument("--shape_influencing_input_names", type=str2list, default=list(),
    #                     help="name list of inputs which influencing other tensors\' shape during inference, like:input1,input2. \
    #                         if set, test_input is required")
    # parser.add_argument("--dynamic", action='store_true',
    #                     help='only valid for onnx model. if set, will automatically set inputs with dyanmic axis \
    #                         as dynamic_shape_input_names and set 1-d inputs as shape_influencing_input_names')
    args = parser.parse_args()
    tpu_mlir_jit.args = args
    tpu_dev = "tpu:0"
    device = torch.device(tpu_dev)
    if args.model == "resnet50":
        import torchvision.models as models
        if args.fp == "fp16":
            input = torch.randn((8, 3, 224, 224),dtype = torch.float16)
            mod = models.resnet50(torch.float16)
        else:
            input = torch.randn((8, 3, 224, 224))
            mod = models.resnet50()
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d[0,0].backward()
        optimizer.step()
    elif args.model == "bert_large":
        from transformers import BertTokenizer, BertModel
        if args.fp == 'fp16':
            mod = BertModel.from_pretrained('bert-large-cased').half()
        else:
            mod = BertModel.from_pretrained('bert-large-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        example_text = 'hello world!'
        bert_input = tokenizer(example_text,padding='max_length',
                            max_length = 10,
                            truncation=True,
                            return_tensors="pt")
        mask = bert_input['attention_mask'].to(device)
        input_id = bert_input['input_ids'].to(device)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        out = model_opt(input_id,mask)
        loss_d = out[0][:, 0, :]  # [batch, 768]
        loss_d[0,0].backward()
        optimizer.step()
    elif args.model == "test_model":
        import torch.nn as nn
        class test_model1(nn.Module):
            def __init__(self, for_train = True):
                super(test_model1, self).__init__()
                self.name = 'test_model1'
                self.conv1 = nn.Conv2d(3, 128, 7, 2, 3, bias=False)
                # self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
                self.relu = nn.ReLU()
                self.for_train = for_train

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                # x = self.conv2(x)
                if self.for_train:
                    x = x.sum()
                return x
        mod = test_model1().half()
        input = torch.randn(8,3,224,224,dtype = torch.float16)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d.backward()
        optimizer.step()
    elif args.model == "sd_lora":
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,DiffusionPipeline,DDPMScheduler,UNet2DConditionModel
        from transformers import CLIPTokenizer
        import torch.nn.functional as F
        from diffusers.models.attention_processor import LoRAAttnProcessor
        from diffusers.loaders import AttnProcsLayers
        import torch.nn as nn
        model_id = "runwayml/stable-diffusion-v1-5"
        cliptokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        num_train_timesteps = 1000
        beta_start = 0.0001
        beta_end = 0.02
        if args.fp == "fp16":
            mod = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            vae_input = torch.randn(1,3,512,512,dtype = torch.float16).to(device)
            noise = torch.randn(1,4,64,64,dtype = torch.float16).to(device)
            unet_input = torch.randn(1,4,64,64,dtype = torch.float16).to(device)
            encoder_hidden_state = torch.randn(1,77,768,dtype = torch.float16).to(device)
            # beta = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            # alpha = 1.0 - beta
            # alphas_cumprod = torch.cumprod(alpha, dim=0).to(device,dtype=torch.float16)
            alphas_cumprod = torch.tensor(1).to(device,dtype=torch.float16)
        else:
            mod = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32) #float16
            unet_input = torch.randn(1,4,64,64).to(device)
            vae_input = torch.randn(1,3,512,512).to(device)
            noise = torch.randn(1,4,64,64).to(device)
            encoder_hidden_state = torch.randn(1,77,768).to(device)
            beta = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            alpha = 1.0 - beta
            alphas_cumprod = torch.cumprod(alpha, dim=0).to(device)
        timesteps = torch.tensor(1).to(device)
        timesteps = timesteps.long()
        example_text = 'hello world'
        clip_input = cliptokenizer(example_text,padding='max_length',
                            max_length = 77,
                            truncation=True,#True
                            return_tensors="pt")
        # mask = clip_input['attention_mask'].to(device)
        input_id = clip_input['input_ids'].to(device)

        unet = mod.unet.to(device)
        vae = mod.vae.to(device)
        text_encoder = mod.text_encoder.to(device)

        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        ########## set lora_layers ############
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
                rank=args.rank,
            )
        unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors)

        unet = unet.to(device)

        optimizer = torch.optim.SGD(lora_layers.parameters(), lr=0.01)
        optimizer.zero_grad()

        class noise_module(nn.Module):
            def __init__(self):
                super(noise_module,self).__init__()
            def forward(self,original_samples,noise,timesteps,alphas_cumprod):
                alphas_cumprod = alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
                timesteps = timesteps.to(original_samples.device)

                # sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
                sqrt_alpha_prod = alphas_cumprod ** 0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                # sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod) ** 0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

                noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
                return noisy_samples

        class sd_module(nn.Module):
            def __init__(self,unet,vae,text_encoder,noise_scheduler,tokenizer):
                super(sd_module, self).__init__()
                self.unet = unet
                self.vae = vae
                self.text_encoder = text_encoder
                self.noise_scheduler = noise_scheduler
                self.mse_loss = nn.MSELoss(reduction='sum')
                self.tokenizer = tokenizer
                self.seq_len = 77

            def forward(self,noise,vae_input,timesteps,input_id,alphas_cumprod):
                latents = self.vae.encode(vae_input).latent_dist.sample()
                noisy_latents = self.noise_scheduler(latents.to(device), noise, timesteps,alphas_cumprod)
                encoder_hidden_state = self.text_encoder(input_id)[0]
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_state)[0]
                loss = self.mse_loss(model_pred.float(), noise.float())
                return loss
        noise_scheduler = noise_module()
        sd = sd_module(unet,vae,text_encoder,noise_scheduler,cliptokenizer)
        sd_opt = torch.compile(sd,backend=aot_backend)
        loss = sd_opt(noise,vae_input,timesteps,input_id,alphas_cumprod)
        loss.backward()
        optimizer.step()
    elif args.model == "yolo":
        input = torch.randn((1,3,640,640))
        from nets.yolo import YoloBody
        anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_classes = 80
        phi = 's'
        mod = YoloBody(anchors_mask, num_classes, phi)
        if args.fp == "fp16":
            input = torch.randn((1,3,640,640),dtype = torch.float16)
            # input = input.half()
            mod = mod.half()
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d[0][0,0,0,0].backward()
        optimizer.step()
    elif args.model == "test_runtime":
        import torch.nn as nn
        class test_runtime(nn.Module):
            def __init__(self):
                super(test_runtime, self).__init__()
                self.p = nn.Linear(3,3,bias=False)

            def forward(self, x):
                x = self.p(x)
                return x
        mod = test_runtime()
        input = torch.randn(10,3)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        res = model_opt(input_d)
        loss_d = torch.sum(res)
        loss_d.backward()
        optimizer.step()
    elif args.model_name == "test_compiler":
        class test_model(nn.Module):
            def __init__(self, for_train = True):
                super(test_model, self).__init__()
                self.linear = nn.Linear(3,3)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                return x
        mod = test_model()
        input = torch.randn(3,3,dtype = torch.float32)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d[0,0].backward()
        optimizer.step()
