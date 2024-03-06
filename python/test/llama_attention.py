import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
tpu_device = "tpu:0"
torch.set_printoptions(profile="full")


def SophLlamaAttentionDecode(q, k, v, k_cache, v_cache, cos, sin, input_lengths, save_slots, fetch_slots, mask, slots_size, mask_size, block_size, C):

    attn_out = torch.empty_like(q).to(q.device)
    torch.ops.my_ops.llama_attention( attn_out, q, k, v, k_cache, v_cache, 
                                      cos, sin, input_lengths, save_slots, fetch_slots, 
                                      mask, slots_size, mask_size, block_size, C, 3)
    return attn_out.cpu()

def SophLlamaAttentionPrefill(q, k, v, k_cache, v_cache, cos, sin, input_lengths, save_slots, fetch_slots, mask, slots_size, mask_size, block_size, C):

    attn_out = torch.empty_like(q).to(q.device)
    torch.ops.my_ops.llama_attention( attn_out, q, k, v, k_cache, v_cache, 
                                      cos, sin, input_lengths, save_slots, fetch_slots, 
                                      mask, slots_size, mask_size, block_size, C, 2)
    return attn_out.cpu()


def case(isPrefill):
    if isPrefill:
        data = torch.load("../../attention/PrefillTPU.pth", map_location="cpu")
    else:
        data = torch.load("../../attention/DecodeCPU.pth", map_location="cpu")

    query = data['query']
    key = data['key']
    value = data['value']
    k_cache = data['k_cache']
    v_cache = data['v_cache']
    cos = data['cos']
    sin = data['sin']
    input_lengths = data['input_lengths']
    save_slots = data['save_slots']
    fetch_slots = data['fetch_slots']
    slots_size = data['slots_size']
    mask = data['mask']
    mask_size = data['mask_size']
    block_size = data['block_size']
    softmax_scale = data['softmax_scale']

    attention_out = data['attention_out']

    fetch_slots = fetch_slots.to(tpu_device) if fetch_slots is not None else None
    mask = mask.to(tpu_device) if mask is not None else None

    if isPrefill:
        attn_out = SophLlamaAttentionPrefill(query.to(tpu_device), key.to(tpu_device), value.to(tpu_device), k_cache.to(tpu_device), v_cache.to(tpu_device), 
                                         cos.to(tpu_device), sin.to(tpu_device), input_lengths.to(tpu_device), 
                                         save_slots.to(tpu_device), fetch_slots, mask, 
                                         slots_size, mask_size, block_size, softmax_scale)
    else:
        attn_out =  SophLlamaAttentionDecode(query.to(tpu_device), key.to(tpu_device), value.to(tpu_device), k_cache.to(tpu_device), v_cache.to(tpu_device), 
                                            cos.to(tpu_device), sin.to(tpu_device), input_lengths.to(tpu_device), 
                                            save_slots.to(tpu_device), fetch_slots, mask, 
                                            slots_size, mask_size, block_size, softmax_scale)
    
    print(f"max diff: {torch.max(torch.abs(attention_out - attn_out))}")


def case_contiguous():
    head_size= 128
    num_heads = 8
    num_key_value_heads = 1
    batches_num = 16
    block_size = 16

    qkv = torch.randn((batches_num, (num_heads+2*num_key_value_heads)*head_size), dtype=torch.float16).to(tpu_device)
    query, key, value = qkv.split(
        [
            head_size * num_heads,
            head_size * num_key_value_heads,
            head_size * num_key_value_heads,
        ],
        dim=1,
    )
    query = query.view(-1, num_heads, head_size)
    kv_view = lambda x : x.view(-1, num_key_value_heads, head_size)
    key = kv_view(key)
    value = kv_view(value)

    batches_num = 16
    sequence_length = 4096
    head_size = 128
    num_heads = 8
    num_key_value_heads = 1

    
    slots = torch.tensor([sequence_length*i for i in range(1, batches_num+1)], dtype=torch.int32, device=tpu_device)
    block_tables = torch.arange(sequence_length, dtype=torch.int32, device=tpu_device).view(batches_num, 256)

    k_cache = torch.randn((4096, block_size, num_key_value_heads, head_size), dtype=torch.float16, device=tpu_device)
    v_cache = torch.randn((4096, block_size, num_key_value_heads, head_size), dtype=torch.float16, device=tpu_device)

    cos = torch.randn((batches_num, 1, head_size), dtype=torch.float16, device=tpu_device)
    sin = torch.randn((batches_num, 1, head_size), dtype=torch.float16, device=tpu_device)
    mask = None
    mask_size = 0
    softmax_scale = head_size**-0.5
    input_lengths = torch.tensor([sequence_length-1 for i in range(16)], dtype=torch.int32, device=tpu_device)

    attention_out = torch.empty_like(query)
    save_slots = slots.unsqueeze(1)
    fetch_slots = block_tables * block_size

    torch.ops.my_ops.llama_attention(attention_out, query, key, value, k_cache, v_cache, 
                                        cos, sin, input_lengths, save_slots, fetch_slots, mask,
                                        block_tables.size(1), mask_size, block_size, softmax_scale, 3)
    
    print(attention_out.shape)

def case_llama_attention_1684x_prefill():
    num_heads = 32
    num_kv_heads = 32
    hidden_size = 4096

    head_size = hidden_size // num_heads
    softmax_scale = head_size**-0.5


    # prefill
    input_lengths = torch.tensor([4096, 4096, 4096, 4096], dtype=torch.int32)
    Ntotal = input_lengths.sum().item()
    max_s = input_lengths.max().item()
    batches = len(input_lengths)

    query = torch.randn((Ntotal, num_heads, head_size), dtype=torch.float16, device=device)
    key = torch.randn((Ntotal,  num_heads, head_size), dtype=torch.float16, device=device)
    value = torch.randn((Ntotal,  num_heads, head_size), dtype=torch.float16, device=device)
    # import pdb; pdb.set_trace()

    block_size = 16
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)

    cu_seqlen_prefill = torch.zeros((batches + 1, ), dtype=torch.int32, device=device)
    slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)

    block_id = 0
    for i in range(batches):
        cu_seqlen_prefill[i + 1] = cu_seqlen_prefill[i] + input_lengths[i]
        need_block_num = (input_lengths[i] + block_size - 1) // block_size
        temp = torch.arange(input_lengths[i], dtype=torch.int32)+(block_id*block_size)
        slots[cu_seqlen_prefill[i] : cu_seqlen_prefill[i + 1]] = temp.to(device)
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    k_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)
    v_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)

    attention_out = torch.empty_like(query)

    # import pdb; pdb.set_trace()

    print('begin')
    if cu_seqlen_prefill is not None: 
        mask = torch.triu(torch.full((max_s, max_s), float('-inf'), dtype=torch.float16), diagonal=1).to(query.device)
        save_slots = block_tables.cpu() * block_size
        save_slots = save_slots.to(device)
        fetch_slots = None
        torch.ops.my_ops.llama_attention(attention_out, query, key, value, k_cache, v_cache, 
                                        None, None, input_lengths.to(device), save_slots, fetch_slots, mask,
                                        block_tables.size(1), max_s, block_size, softmax_scale, 2)
    print(attention_out.shape)

    print("done")


def case_llama_attention_1684x_decode():
    num_heads = 32
    num_kv_heads =  32
    head_size = 128
    softmax_scale = head_size**-0.5

    input_lengths = torch.tensor([2 for _ in range(16)], dtype=torch.int32).to(device)
    batches = len(input_lengths)
    Ntotal = batches
    max_s = max(input_lengths).item()

    block_size = 16
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)

    cu_seqlen_prefill = None
    slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)

    block_id = 0
    for i in range(batches):
        need_block_num = (input_lengths[i].item() + block_size - 1) // block_size
        slots[i] = block_id * block_size + input_lengths[i]-1
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    kv_cache = [(torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device),
                  torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)) for _ in range(1)]

    input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32).to(device)
    position_ids = input_lengths.clone().detach()

    query = torch.randn((Ntotal, num_heads, head_size), dtype=torch.float16, device=device)
    key = torch.randn((Ntotal, num_heads, head_size), dtype=torch.float16, device=device)
    value = torch.randn((Ntotal, num_heads, head_size), dtype=torch.float16, device=device)

    mask = None
    save_slots = slots.unsqueeze(1)
    fetch_slots = block_tables.cpu() * block_size
    fetch_slots = fetch_slots.to(query.device)

    attention_out = torch.empty_like(query)

    torch.ops.my_ops.llama_attention(attention_out, query, key, value, kv_cache[0][0], kv_cache[0][1], 
                                         None, None, input_lengths, save_slots, fetch_slots, mask,
                                         block_tables.size(1), max_s, block_size, softmax_scale, 3)
    print('done')

def case_llama_attention_2260_decode():
    num_heads = 64
    num_kv_heads =  8
    head_size = 128
    softmax_scale = head_size**-0.5

    input_lengths = torch.tensor([2 for _ in range(4)], dtype=torch.int32).to(device)
    batches = len(input_lengths)
    Ntotal = batches
    max_s = max(input_lengths).item()

    block_size = 16
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)

    cu_seqlen_prefill = None
    slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)

    block_id = 0
    for i in range(batches):
        need_block_num = (input_lengths[i].item() + block_size - 1) // block_size
        slots[i] = block_id * block_size + input_lengths[i]-1
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    kv_cache = [(torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device),
                  torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)) for _ in range(1)]

    input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32).to(device)
    position_ids = input_lengths.clone().detach()

    query = torch.randn((Ntotal, num_heads, head_size), dtype=torch.float16, device=device)
    key = torch.randn((Ntotal, num_kv_heads, head_size), dtype=torch.float16, device=device)
    value = torch.randn((Ntotal, num_kv_heads, head_size), dtype=torch.float16, device=device)

    mask = None
    save_slots = slots.unsqueeze(1)
    fetch_slots = block_tables.cpu() * block_size
    fetch_slots = fetch_slots.to(query.device)

    attention_out = torch.empty_like(query)

    torch.ops.my_ops.llama_attention(attention_out, query, key, value, kv_cache[0][0], kv_cache[0][1], 
                                         None, None, input_lengths, save_slots, fetch_slots, mask,
                                         block_tables.size(1), max_s, block_size, softmax_scale, 3)
    print('done')

if __name__ == "__main__":
    # case(False)
    # case_contiguous()
    # case_llama_attention_1684x_prefill()
    # case_llama_attention_1684x_decode()
    case_llama_attention_2260_decode()
