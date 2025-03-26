import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
import time

import torch_tpu
import torch_tpu.tpu
torch.manual_seed(200)
device = "tpu:0"

def gen_rope(Ntotal, head_size):
    theta = torch.arange(0, head_size, 2).float() / head_size
    theta = 1. / 10000 ** theta
    theta = theta.view(1, -1).squeeze()  # (1, d/2)
    seq_idx = torch.arange(Ntotal, dtype=theta.dtype)
    seq_idx = seq_idx.view(-1, 1).squeeze()  # (seq_len, 1)
    idx_theta = torch.einsum('i,j->ij', seq_idx, theta)  # (seq_len, d/2)
    idx_theta = idx_theta.view(idx_theta.shape[0],1,-1)
    idx_theta = idx_theta.repeat(1,1,2)
    return idx_theta

def rotate_every_two(x):
    x_swapped = torch.cat((-x[:, :, (int)(x.shape[-1]/2):], x[:, :, :(int)(x.shape[-1]/2)]), dim=-1)
    return x_swapped

def case_mla_2260_decode():
    head = 16
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = 192
    v_head_dim = 128
    softmax_scale = qk_head_dim**-0.5

    input_lengths = torch.tensor([4096 for _ in range(1)], dtype=torch.int32).to("cpu")
    batches = len(input_lengths)
    max_seq_len = max(input_lengths).item()

    seqlen = 4096
    Ntotal = seqlen * batches

    kv_cache = torch.randn((batches, max_seq_len * 4, kv_lora_rank), dtype=torch.bfloat16, device=device)
    pe_cache = torch.randn((batches, max_seq_len * 4, qk_rope_head_dim), dtype=torch.bfloat16, device=device)

    Q = torch.randn((batches, 1, q_lora_rank), dtype=torch.bfloat16, device=device)
    KV = torch.randn((batches, 1, kv_lora_rank), dtype=torch.bfloat16, device=device)
    PE = torch.randn((batches, 1, qk_rope_head_dim), dtype=torch.bfloat16, device=device)

    WUQ = torch.randn((head * qk_head_dim, q_lora_rank), dtype=torch.bfloat16, device=device)
    WUKV = torch.randn((head * (qk_nope_head_dim + v_head_dim), kv_lora_rank), dtype=torch.bfloat16, device=device)

    idx_theta = gen_rope(Ntotal, qk_rope_head_dim).bfloat16().to(device) #[seq_len,1,head_size]
    cos = torch.cos(idx_theta[:,0,:]).view(idx_theta.shape[0],1,-1).contiguous().to(device)
    sin = torch.sin(idx_theta[:,0,:]).view(idx_theta.shape[0],1,-1).contiguous().to(device)

    mask = None

    Out = torch.empty((batches, 1, head, v_head_dim), dtype=torch.bfloat16, device=device)


    torch.ops.my_ops.mla(Out, Q, KV, PE, WUQ, WUKV, kv_cache, pe_cache,
                        cos, sin, mask, input_lengths, head,
                        q_lora_rank, kv_lora_rank, qk_nope_head_dim,
                        qk_rope_head_dim, v_head_dim, max_seq_len, softmax_scale, 1)


    # step 1: Q upper
    Q = Q.view(Q.shape[0], -1)
    Q = torch.matmul(Q, WUQ.t())
    Q = Q.view(batches, 1, head, qk_head_dim)
    q_nope, q_pe = torch.split(Q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)

    # step 2: Q RoPE
    q_pe = q_pe.view(batches, head, qk_rope_head_dim)
    q_pe = q_pe * torch.cos(idx_theta[0]) + rotate_every_two(q_pe) * torch.sin(idx_theta[0])
    q_pe = q_pe.view(batches, 1, head, qk_rope_head_dim)

    # step 3: Q_nope * WUKV[:, :qk_nope_head_dim]
    WUKV = WUKV.view(head, -1, kv_lora_rank)
    q_nope = torch.einsum("bshd,hdc->bshc", q_nope, WUKV[:, :qk_nope_head_dim])

    # step 4：QK
    start_pos = max_seq_len
    end_pos = start_pos + 1

    kv_cache[:batches, start_pos:end_pos] = KV
    PE = PE * torch.cos(idx_theta[0]) + rotate_every_two(PE) * torch.sin(idx_theta[0])
    pe_cache[:batches, start_pos:end_pos] = PE
    scores = (torch.einsum("bshc,btc->bsht", q_nope, kv_cache[:batches, :end_pos]) + 
                      torch.einsum("bshr,btr->bsht", q_pe, pe_cache[:batches, :end_pos])) * softmax_scale 

    # step5 softmax
    scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(Q)
    
    # step6 QK * KVcache
    x = torch.einsum("bsht,btc->bshc", scores, kv_cache[:batches, :end_pos]) 

    # step7 QKV * WUKV[:, -self.v_head_dim:]
    x = torch.einsum("bshc,hdc->bshd", x, WUKV[:, -v_head_dim:])

    print(x.cpu())
    print(Out.cpu())
    diff = Out.cpu() - x.cpu()
    print(torch.max(torch.abs(diff)))

    import torch.nn.functional as F
    tensor1_flat = Out.cpu().flatten()
    tensor2_flat = x.cpu().flatten()
    cosine_similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=0)
    cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
    print("{:.5f}".format(cosine_similarity.item()))
    
        
if __name__ == "__main__":
    case_mla_2260_decode()