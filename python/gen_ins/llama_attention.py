import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
import os, shutil, glob

torch.set_num_threads(1)

from utils import DumpIns
DI = DumpIns()

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
tpu_device = "tpu:0"
torch.set_printoptions(profile="full")

def pre_gen_inst(dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)
    print(os.getcwd())
    os.chdir(dst)
    print(os.getcwd())

def post_gen_inst():
    if os.path.isdir("prepare"):
        shutil.rmtree("prepare")
    
    reg_files = glob.glob(f'*reg*.txt')
    for f in reg_files:
        shutil.copy(f, "attn")

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
    post_gen_inst()
    print('done')

def case_llama_attention_2260_prefill(num_heads=64, num_kv_heads=8, hidden_size=8192, batch=16, TP=8, seq_len=4096):
    head_size = hidden_size // num_heads
    softmax_scale = head_size**-0.5

    num_heads = num_heads // TP
    num_kv_heads = num_kv_heads // TP
    
    DI.dump("prepare")

    # prefill
    input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32)
    Ntotal = input_lengths.sum().item()
    max_s = input_lengths.max().item()
    batches = len(input_lengths)

    # create tensor in cpu side to speed up
    # qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16).to(device)
    qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16)
    query, key, value = qkv.split(
        [
            head_size * num_heads,
            head_size * num_kv_heads,
            head_size * num_kv_heads,
        ],
        dim=1,
    )
    query = query.view(-1, num_heads, head_size)
    kv_view = lambda x : x.view(-1, num_kv_heads, head_size)
    key = kv_view(key)
    value = kv_view(value)

    block_size = 16
    #block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32)

    #cu_seqlen_prefill = torch.zeros((batches + 1, ), dtype=torch.int32, device=device)
    cu_seqlen_prefill = torch.zeros((batches + 1, ), dtype=torch.int32)
    #slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)

    block_id = 0
    for i in range(batches):
        cu_seqlen_prefill[i + 1] = cu_seqlen_prefill[i] + input_lengths[i]
        need_block_num = (input_lengths[i] + block_size - 1) // block_size
        temp = torch.arange(input_lengths[i], dtype=torch.int32)+(block_id*block_size)
        #slots[cu_seqlen_prefill[i] : cu_seqlen_prefill[i + 1]] = temp.to(device)
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    #k_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)
    #v_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)
    k_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16)
    v_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16)

    attention_out = torch.empty_like(query)

    print('begin')
    DI.dump("attn")
    if cu_seqlen_prefill is not None:
        # to device
        qkv = qkv.to(device)
        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_tables = block_tables.to(device)
        cu_seqlen_prefill = cu_seqlen_prefill.to(device)
        k_cache = k_cache.to(device)
        v_cache = v_cache.to(device)
        attention_out = attention_out.to(device)

        mask = torch.triu(torch.full((max_s, max_s), float('-inf'), dtype=torch.float16), diagonal=1).to(query.device)
        save_slots = block_tables.cpu() * block_size
        save_slots = save_slots.to(device)
        fetch_slots = None
        torch.ops.my_ops.llama_attention(attention_out, query, key, value, k_cache, v_cache, 
                                        None, None, input_lengths.to(device), save_slots, fetch_slots, mask,
                                        block_tables.size(1), max_s, block_size, softmax_scale, 2)

    post_gen_inst()
    print('done')

def case_llama_attention_2260_decode(num_heads=64, num_kv_heads=8, hidden_size=8192, batch=16, TP=8, seq_len=4096):
    # LLama 2 70b TP8, batch 16
    head_size = hidden_size // num_heads
    softmax_scale = head_size**-0.5

    DI.dump("prepare")

    num_heads = num_heads // TP
    num_kv_heads = num_kv_heads // TP

    #input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32).to(device)
    input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32)
    batches = len(input_lengths)
    Ntotal = batches
    max_s = max(input_lengths).item()

    block_size = 16
    #block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32)

    cu_seqlen_prefill = None
    #slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)
    slots = torch.zeros((Ntotal, ), dtype=torch.int32)

    block_id = 0
    for i in range(batches):
        need_block_num = (input_lengths[i].item() + block_size - 1) // block_size
        slots[i] = block_id * block_size + input_lengths[i]-1
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    kv_cache = [(torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device),
                  torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)) for _ in range(1)]

    #input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32).to(device)
    input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32)
    position_ids = input_lengths.clone().detach()

    #qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16).to(device)
    qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16)

    # to device
    input_lengths = input_lengths.to(device)
    block_tables = block_tables.to(device)
    slots = slots.to(device)
    input_ids = input_ids.to(device)
    qkv = qkv.to(device)

    DI.dump("view")
    query, key, value = qkv.split(
        [
            head_size * num_heads,
            head_size * num_kv_heads,
            head_size * num_kv_heads,
        ],
        dim=1,
    )
    query = query.view(-1, num_heads, head_size)
    kv_view = lambda x : x.view(-1, num_kv_heads, head_size)
    key = kv_view(key)
    value = kv_view(value)

    #DI.dump("save_pth")
    #torch.save(qkv, "decode_qkv_cpu.pth")
    #torch.save(query.cpu(), "decode_query_cpu.pth")
    #torch.save(key.cpu(), "decode_key_cpu.pth")
    #torch.save(value.cpu(), "decode_value_cpu.pth")

    mask = None
    save_slots = slots.unsqueeze(1)
    fetch_slots = block_tables.cpu() * block_size
    fetch_slots = fetch_slots.to(query.device)

    attention_out = torch.empty_like(query)

    DI.dump("attn")
    #import pdb; pdb.set_trace()
    torch.ops.my_ops.llama_attention(attention_out, query, key, value, kv_cache[0][0], kv_cache[0][1], 
                                         None, None, input_lengths, save_slots, fetch_slots, mask,
                                         block_tables.size(1), max_s, block_size, softmax_scale, 3)

    #torch.save(attention_out, "decode_attention_out.pth")
    #torch.save(attention_out.cpu(), "decode_attention_out_cpu.pth")
    post_gen_inst()
    print('done')


def case_llama_attention_70b_tp8_2260_prefill():
    # LLama2 70B TP8, batch 16
    num_heads = 64
    num_kv_heads = 8
    hidden_size = 8192

    head_size = hidden_size // num_heads
    softmax_scale = head_size**-0.5

    batch = 16
    TP = 8
    num_heads = num_heads // TP
    num_kv_heads = num_kv_heads // TP
    seq_len = 4096
    
    DI.dump("prepare")

    # prefill
    input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32)
    Ntotal = input_lengths.sum().item()
    max_s = input_lengths.max().item()
    batches = len(input_lengths)

    # create tensor in cpu side to speed up
    # qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16).to(device)
    qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16)
    query, key, value = qkv.split(
        [
            head_size * num_heads,
            head_size * num_kv_heads,
            head_size * num_kv_heads,
        ],
        dim=1,
    )
    query = query.view(-1, num_heads, head_size)
    kv_view = lambda x : x.view(-1, num_kv_heads, head_size)
    key = kv_view(key)
    value = kv_view(value)

    block_size = 16
    #block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32)

    #cu_seqlen_prefill = torch.zeros((batches + 1, ), dtype=torch.int32, device=device)
    cu_seqlen_prefill = torch.zeros((batches + 1, ), dtype=torch.int32)
    #slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)

    block_id = 0
    for i in range(batches):
        cu_seqlen_prefill[i + 1] = cu_seqlen_prefill[i] + input_lengths[i]
        need_block_num = (input_lengths[i] + block_size - 1) // block_size
        temp = torch.arange(input_lengths[i], dtype=torch.int32)+(block_id*block_size)
        #slots[cu_seqlen_prefill[i] : cu_seqlen_prefill[i + 1]] = temp.to(device)
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    #k_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)
    #v_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)
    k_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16)
    v_cache = torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16)

    attention_out = torch.empty_like(query)

    print('begin')
    DI.dump("attn")
    if cu_seqlen_prefill is not None:
        # to device
        qkv = qkv.to(device)
        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        block_tables = block_tables.to(device)
        cu_seqlen_prefill = cu_seqlen_prefill.to(device)
        k_cache = k_cache.to(device)
        v_cache = v_cache.to(device)
        attention_out = attention_out.to(device)

        mask = torch.triu(torch.full((max_s, max_s), float('-inf'), dtype=torch.float16), diagonal=1).to(query.device)
        save_slots = block_tables.cpu() * block_size
        save_slots = save_slots.to(device)
        fetch_slots = None
        torch.ops.my_ops.llama_attention(attention_out, query, key, value, k_cache, v_cache, 
                                        None, None, input_lengths.to(device), save_slots, fetch_slots, mask,
                                        block_tables.size(1), max_s, block_size, softmax_scale, 2)

    post_gen_inst()
    print('done')

def case_llama_attention_70b_tp8_2260_decode():
    # LLama 2 70b TP8, batch 16
    num_heads = 64
    num_kv_heads =  8
    head_size = 128
    softmax_scale = head_size**-0.5

    DI.dump("prepare")

    batch = 16
    TP = 8
    num_heads = num_heads // TP
    num_kv_heads = num_kv_heads // TP
    seq_len = 4096

    #input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32).to(device)
    input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32)
    batches = len(input_lengths)
    Ntotal = batches
    max_s = max(input_lengths).item()

    block_size = 16
    #block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32)

    cu_seqlen_prefill = None
    #slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)
    slots = torch.zeros((Ntotal, ), dtype=torch.int32)

    block_id = 0
    for i in range(batches):
        need_block_num = (input_lengths[i].item() + block_size - 1) // block_size
        slots[i] = block_id * block_size + input_lengths[i]-1
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    kv_cache = [(torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device),
                  torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)) for _ in range(1)]

    #input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32).to(device)
    input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32)
    position_ids = input_lengths.clone().detach()

    #qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16).to(device)
    qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16)

    # to device
    input_lengths = input_lengths.to(device)
    block_tables = block_tables.to(device)
    slots = slots.to(device)
    input_ids = input_ids.to(device)
    qkv = qkv.to(device)

    DI.dump("view")
    query, key, value = qkv.split(
        [
            head_size * num_heads,
            head_size * num_kv_heads,
            head_size * num_kv_heads,
        ],
        dim=1,
    )
    query = query.view(-1, num_heads, head_size)
    kv_view = lambda x : x.view(-1, num_kv_heads, head_size)
    key = kv_view(key)
    value = kv_view(value)

    #DI.dump("save_pth")
    #torch.save(qkv, "decode_qkv_cpu.pth")
    #torch.save(query.cpu(), "decode_query_cpu.pth")
    #torch.save(key.cpu(), "decode_key_cpu.pth")
    #torch.save(value.cpu(), "decode_value_cpu.pth")

    mask = None
    save_slots = slots.unsqueeze(1)
    fetch_slots = block_tables.cpu() * block_size
    fetch_slots = fetch_slots.to(query.device)

    attention_out = torch.empty_like(query)

    DI.dump("attn")
    torch.ops.my_ops.llama_attention(attention_out, query, key, value, kv_cache[0][0], kv_cache[0][1], 
                                         None, None, input_lengths, save_slots, fetch_slots, mask,
                                         block_tables.size(1), max_s, block_size, softmax_scale, 3)

    #torch.save(attention_out, "decode_attention_out.pth")
    #torch.save(attention_out.cpu(), "decode_attention_out_cpu.pth")
    post_gen_inst()
    print('done')

def case_llama_7b_attention_2260_decode():
    num_heads = 32
    num_kv_heads =  32
    head_size = 128
    softmax_scale = head_size**-0.5

    DI.dump("prepare")
    #import pdb; pdb.set_trace()

    batch = 16
    TP = 1
    num_heads = num_heads // TP
    num_kv_heads = num_kv_heads // TP
    seq_len = 4096

    # use cpu to speed up
    #input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32).to(device)
    input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32)
    batches = len(input_lengths)
    Ntotal = batches
    max_s = max(input_lengths).item()

    block_size = 16
    #block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32, device=device)
    block_tables = torch.zeros((batches, (max_s + block_size - 1) // block_size), dtype=torch.int32)

    cu_seqlen_prefill = None
    #slots = torch.zeros((Ntotal, ), dtype=torch.int32, device=device)
    slots = torch.zeros((Ntotal, ), dtype=torch.int32)

    block_id = 0
    for i in range(batches):
        need_block_num = (input_lengths[i].item() + block_size - 1) // block_size
        slots[i] = block_id * block_size + input_lengths[i]-1
        for j in range(need_block_num):
            block_tables[i][j] = block_id
            block_id += 1

    kv_cache = [(torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device),
                  torch.randn((block_id, block_size, num_kv_heads, head_size), dtype=torch.float16, device=device)) for _ in range(1)]

    #input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32).to(device)
    input_ids = torch.randint(0, 32000, (Ntotal, ), dtype=torch.int32)
    position_ids = input_lengths.clone().detach()

    #qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16).to(device)
    qkv = torch.randn((Ntotal, (num_heads+2*num_kv_heads)*head_size), dtype=torch.float16)
    query, key, value = qkv.split(
        [
            head_size * num_heads,
            head_size * num_kv_heads,
            head_size * num_kv_heads,
        ],
        dim=1,
    )
    query = query.view(-1, num_heads, head_size)
    kv_view = lambda x : x.view(-1, num_kv_heads, head_size)
    key = kv_view(key)
    value = kv_view(value)

    #torch.save(qkv.cpu(), "decode_qkv_cpu.pth")
    #torch.save(query.cpu(), "decode_query_cpu.pth")
    #torch.save(key.cpu(), "decode_key_cpu.pth")
    #torch.save(value.cpu(), "decode_value_cpu.pth")

    mask = None
    save_slots = slots.unsqueeze(1)
    fetch_slots = block_tables.cpu() * block_size
    #fetch_slots = fetch_slots.to(query.device)
    fetch_slots = fetch_slots.to(query)

    attention_out = torch.empty_like(query)

    # to device
    input_lengths = input_lengths.to(device)
    block_tables = block_tables.to(device)
    slots = slots.to(device)
    fetch_slots = fetch_slots.to(device)
    save_slots = save_slots.to(device)
    intpu_ids = input_ids.to(device)
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    attention_out = attention_out.to(device)

    DI.dump("attn")
    torch.ops.my_ops.llama_attention(attention_out, query, key, value, kv_cache[0][0], kv_cache[0][1], 
                                         None, None, input_lengths, save_slots, fetch_slots, mask,
                                         block_tables.size(1), max_s, block_size, softmax_scale, 3)

    #torch.save(attention_out, "decode_attention_out.pth")
    #torch.save(attention_out.cpu(), "decode_attention_out_cpu.pth")
    post_gen_inst()
    print('done')

def case_llama_attention_70b_tp8_2260_prefill_2():
    pre_gen_inst("llama-70b-tp8-b16-prefill")
    case_llama_attention_2260_prefill(num_heads=64,
                                      num_kv_heads=8,
                                      hidden_size=8192,
                                      batch=16,
                                      TP=8,
                                      seq_len=4096)

def case_llama_attention_70b_tp8_2260_decode_2():
    pre_gen_inst("llama-70b-tp8-b16-decode")
    case_llama_attention_2260_decode(num_heads=64,
                                     num_kv_heads=8,
                                     hidden_size=8192,
                                     batch=16,
                                     TP=8,
                                     seq_len=4096)

def case_llama_attention_7b_2260_prefill():
    pre_gen_inst("llama-7b-tp1-b16-prefill")
    case_llama_attention_2260_prefill(num_heads=32,
                                      num_kv_heads=32,
                                      hidden_size=4096,
                                      batch=16,
                                      TP=1,
                                      seq_len=4096)

def case_llama_7b_attention_2260_decode_2():
    pre_gen_inst("llama-7b-tp1-b16-decode")
    case_llama_attention_2260_decode(num_heads=32,
                                     num_kv_heads=32,
                                     hidden_size=4096,
                                     batch=16,
                                     TP=1,
                                     seq_len=4096)

def showUsage():
    print("Usage:")
    print("  0: llama 70b TP8 batch 16 prefill")
    print("  1: llama 70b TP8 batch 16 decode")
    print("  2: llama 7b TP1 batch 16 prefill")
    print("  3: llama 7b TP1 batch 16 decode")

if __name__ == "__main__":
    if (len(sys.argv)) < 2:
        showUsage()
        exit()

    act = int(sys.argv[1])

    if act == 0:
        case_llama_attention_70b_tp8_2260_prefill_2()
    elif act == 1:
        case_llama_attention_70b_tp8_2260_decode_2()
    elif act == 2:
        case_llama_attention_7b_2260_prefill()
    elif act == 3:
        case_llama_7b_attention_2260_decode_2()
    else:
        print(f"unexpected {act}")
