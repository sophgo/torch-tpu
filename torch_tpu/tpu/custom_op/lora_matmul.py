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
        torch.ops.my_ops.lora_matmul_forward(x,
                                        loraA,
                                        loraB,
                                        weight,
                                        output,
                                        scale)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        x, loraA, loraB, weight = ctx.saved_tensors
        grad_output_half = grad_output.half()
        grad_loraA = None
        grad_loraB = None
        weight_t = weight.contiguous()
        loraA_t = loraA.contiguous()
        loraB_t = loraB.contiguous()
        x_t = x.transpose(-1, -2).contiguous()
        if ctx.scale == 0:
            grad_input = torch.matmul(grad_output_half, weight_t)
        else:
            grad_input = torch.matmul(grad_output_half, (weight_t + 1 / ctx.scale * torch.matmul(loraB_t, loraA_t)))
            grad_loraA = 1 / ctx.scale * torch.matmul(torch.matmul(x_t, grad_output_half), loraB_t)
            grad_loraB = 1 / ctx.scale * torch.matmul(torch.matmul(loraA_t, x_t), grad_output_half)
        return grad_input.float(), grad_loraA, grad_loraB, None, None, None

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

        weight = peft_lora_module.base_layer.weight.data.contiguous().half().to(device)
        loraA_data = lora_A.weight.data.contiguous().half().to(device)
        loraB_data = lora_B.weight.data.contiguous().half().to(device)
        newlora_block = LoraMatmulBlock(lora_A.in_features, lora_B.out_features, weight,
                                        lora_A=loraA_data,lora_B=loraB_data,
                                        rank=lora_config.r, alpha=lora_config.lora_alpha,
                                        dropout_rate=lora_config.lora_dropout).to(device)
        setattr(parent, target_name, newlora_block)
    if not exist_target_modules:
        raise ValueError(f"lora_model can not match lora_config.target_modules({lora_config.target_modules})")