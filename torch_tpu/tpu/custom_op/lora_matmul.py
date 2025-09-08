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

        # lora A: (r, i), lora B: (o,r), weight: (o,i)
        # x: (b*s, i), output: (b*s, o), grad_output: (b*s, o)
        x_2d = x.view(-1, x.shape[-1]).contiguous()
        grad_out_2d = grad_output.view(-1, grad_output.shape[-1]).contiguous()
        grad_input = torch.matmul(grad_out_2d, weight) + \
            scale * grad_out_2d.matmul(loraB).matmul(loraA)
        grad_input = grad_input.view(x.shape)
        grad_loraA = scale * torch.matmul((torch.matmul(grad_out_2d, loraB)).t(), x_2d)
        grad_loraB = scale * torch.matmul(grad_out_2d.t(), x_2d.matmul(loraA.t()))

        return grad_input, grad_loraA, grad_loraB, None, None

class LoraMatmulBlock(nn.Module):
    def __init__(self, in_feature, out_feature, weight_tensor, lora_A_tensor, lora_B_tensor, rank, alpha, dropout_rate) -> None:
        super(LoraMatmulBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.rank = rank
        self.dropout_rate = dropout_rate

        self.weight_linear = nn.Linear(in_feature, out_feature, bias=False)
        self.weight_linear.weight.data = weight_tensor
        self.weight_linear.weight.requires_grad = False

        if self.rank > 0:
            self.lora_A_linear = nn.Linear(in_feature, rank, bias=False)
            self.lora_B_linear = nn.Linear(rank, out_feature, bias=False)
            self.lora_A_linear.weight.data = lora_A_tensor
            self.lora_B_linear.weight.data = lora_B_tensor
            self.lora_A_linear.weight.requires_grad = True
            self.lora_B_linear.weight.requires_grad = True
            self.scale = alpha / rank

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        if self.rank > 0:
            weight = self.weight_linear.weight
            lora_A = self.lora_A_linear.weight
            lora_B = self.lora_B_linear.weight
            return LoraMatmulFunc.apply(self.dropout(x), lora_A, lora_B, weight, self.scale)
        else:
            return self.weight_linear(self.dropout(x))

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
        if weight.dtype not in [torch.float16, torch.bfloat16] or \
            loraA_data.dtype not in [torch.float16, torch.bfloat16] or \
            loraB_data.dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(f"lora_model must be float16 or bfloat16. please check the dtype")

        newlora_block = LoraMatmulBlock(lora_A.in_features, lora_B.out_features,
                                        weight_tensor=weight,
                                        lora_A_tensor=loraA_data,
                                        lora_B_tensor=loraB_data,
                                        rank=lora_config.r,
                                        alpha=lora_config.lora_alpha,
                                        dropout_rate=lora_config.lora_dropout).to(device)

        setattr(parent, target_name, newlora_block)
    if not exist_target_modules:
        raise ValueError(f"lora_model can not match lora_config.target_modules({lora_config.target_modules})")