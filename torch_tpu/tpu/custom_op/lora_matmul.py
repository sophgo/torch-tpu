import torch
import torch.nn as nn
import math
from peft import LoraConfig, TaskType
import peft
import re

class LoraMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,loraA, loraB, weight,scale):
        ctx.save_for_backward(x, loraA, loraB, weight)
        ctx.scale = scale
        output_shape = list(x.shape)
        output_shape[-1] = loraB.shape[0]
        loraA = loraA.unsqueeze(0)#backend op only accept dim==3 of loraA/loraB
        loraB = loraB.unsqueeze(0)
        output = torch.empty(output_shape, dtype = x.dtype, device=x.device)
        # output = x @ w.t + scale * (x @ loraA.t()) @ loraB.t()
        torch.ops.my_ops.lora_matmul_forward(x,
                                        loraA,
                                        loraB,
                                        weight,
                                        output,
                                        scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, loraA, loraB, weight = ctx.saved_tensors
        grad_loraA = None
        grad_loraB = None
        scale = ctx.scale
        if scale == 0:
            grad_input = torch.matmul(grad_output, weight)
        else:
            # lora A: (r, i), lora B: (o,r), weight: (o,i)
            # x: (b*s, i), output: (b*s, o), grad_output: (b*s, o)
            x_2d = x.view(-1, x.shape[-1]).contiguous()
            grad_out_half_2d = grad_output.view(-1, grad_output.shape[-1]).contiguous()
            grad_input = torch.matmul(grad_out_half_2d, weight) + \
                scale * grad_out_half_2d.matmul(loraB).matmul(loraA)
            grad_input = grad_input.view(x.shape)
            grad_loraA = scale * torch.matmul((torch.matmul(grad_out_half_2d, loraB)).t(), x_2d)
            grad_loraB = scale * torch.matmul(grad_out_half_2d.t(), x_2d.matmul(loraA.t()))

        return grad_input, grad_loraA, grad_loraB, None, None

class LoraMatmulBlock(nn.Module):
    def __init__(self, in_feature, out_feature, weight,lora_A,lora_B, rank, alpha, dropout_rate) -> None:
        super(LoraMatmulBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = weight
        self.weight.requires_grad = False
        self.rank = rank
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()
        if self.rank > 0:
            self.loraA = lora_A
            self.loraB = lora_B
            self.scale = alpha / rank
    def forward(self, x):
        return LoraMatmulFunc.apply(self.dropout(x.half()),self.loraA, self.loraB, self.weight, self.scale)

def create_and_replace(lora_model, lora_config:LoraConfig, adapter_name:str = "default"):
    """
    Args:
        model (_type_): only support huggingface transformers models containing peft lora module
        lora_config (LoraConfig): which is same with peft lora config
        adapter_name (str): same with peft lora
    """
    device = lora_model.device
    key_list = [key for key, _ in lora_model.named_modules()]
    exist_target_modules = False
    for key in key_list:
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        elif key in lora_config.target_modules:
            target_module_found = True
        else:
            target_module_found = any(key.endswith(f".{target_key}") for target_key in lora_config.target_modules)

        if not target_module_found:
            continue
        exist_target_modules = True
        peft_lora_module = lora_model.get_submodule(key)
        parent = lora_model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        lora_A = peft_lora_module.lora_A[adapter_name]
        lora_B = peft_lora_module.lora_B[adapter_name]

        weight = peft_lora_module.base_layer.weight.data
        loraA_data = lora_A.weight.data
        loraB_data = lora_B.weight.data
        if weight.dtype != torch.float16 or loraA_data.dtype != torch.float16 or loraB_data.dtype != torch.float16:
            raise ValueError(f"lora_model must be float16. please check the dtype")
        if weight.device != device or loraA_data.device != device or loraB_data.device != device:
            raise ValueError(f"lora_model must be on device {device}. please check the device")
        newlora_block = LoraMatmulBlock(lora_A.in_features, lora_B.out_features, weight,
                                        lora_A=loraA_data,lora_B=loraB_data,
                                        rank=lora_config.r, alpha=lora_config.lora_alpha,
                                        dropout_rate=lora_config.lora_dropout).to(device)
        newlora_block.loraA.requires_grad = True
        newlora_block.loraB.requires_grad = True
        newlora_block.weight.requires_grad = False #force to false

        setattr(parent, target_name, newlora_block)
    if not exist_target_modules:
        raise ValueError(f"lora_model can not match lora_config.target_modules({lora_config.target_modules})")