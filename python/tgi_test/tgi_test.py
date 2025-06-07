"""
README

This file contains the test, perfAI and profile of specific ops(mlp, add, rmsnorm, attn_fc, mmqkv, attention) 
from llama and qwen modles in the process of prefill and decode. 

Usage:
------
1. test 
#w4a16

python python/test/tgi_test.py --model qwen_7b  --case mlp  --batch 1 --tp 2 --w4a16 --test

#prefill

python python/test/tgi_test.py --model qwen_7b  --case mlp  --batch 1 --tp 2 --prefill --test

#decode

python python/test/tgi_test.py --model qwen_7b  --case mlp  --batch 1 --tp 2 --test

2. perfAI
Make ensure that the tpu-train and PerfAI repos are in the same directory level. Then install some third-party dependencies:
pip install -r requirements.txt

#w4a16

python python/test/tgi_test.py --model qwen_7b  --case mlp  --batch 1 --tp 2 --w4a16 --PerfAI 
./run_perfai.sh

#prefill

python python/test/tgi_test.py --model qwen_7b  --case mlp  --batch 1 --tp 2 --prefill --PerfAI 
./run_perfai.sh

#decode

python python/test/tgi_test.py --model qwen_7b  --case mlp  --batch 1 --tp 2 --PerfAI 
./run_perfai.sh

3. profile

Make ensure that TPU-MLIR has been successfully compiled.

python python/test/tgi_test.py --model qwen_7b  --case mlp  --batch 1 --tp 2 --prefill --profile_mode 0
./run_profile.sh

"""

import torch
import torch_tpu
from torch import nn
import torch.nn.functional as F
import argparse
from loguru import logger
import sys, os, shutil
from dataclasses import dataclass
from tgi_modules import *
import time
import os
import math

# current_time_seed = int(time.time())
# torch.manual_seed(current_time_seed)
torch.manual_seed(1000)
torch.set_printoptions(precision=6)

parser = argparse.ArgumentParser(description="tgi test case")
parser.add_argument("--test", action="store_true", help="if test")
parser.add_argument("--PerfAI", action="store_true", help="if PerfAI")
parser.add_argument("--profile_mode", type=int, default=0, help="set profile mode: 0 pmu only, 1 concise cmd, 2 detailed cmd")
parser.add_argument("--prefill", action="store_true", help="if prefill")
parser.add_argument("--w4a16", action="store_true", help="if w4a16")
parser.add_argument("--batch", type=int, default=8, help="set batch size (default: 8)")
parser.add_argument("--tp", type=int, default=1, help="set tp (default: 1)")
parser.add_argument("--seq", type=int, default=4096, help="set context_len in chat mode (default: 4096)")
parser.add_argument(
    "--case",
    choices=["add", "attn_fc", "rmsnorm", "mlp", "mmqkv", "attn"],
    help="ops in llama2-7b",
)
parser.add_argument(
    "--model",
    choices=["llama2_7b", "llama2_70b", "qwen_72b", "qwen_7b","llama3_8b","llama3_70b"],
    help="ops in llama2-7b",
)

args = parser.parse_args()

logger.add("logfile.log")


llama2_7b_cfg = MODEL_CFG(
    HIDDEN_SIZE=4096, 
    INTER_SIZE=11008, 
    HEAD_NUM=32, 
    KV_HEAD_NUM=32,
    VOCAB_SIZE=32000
)
llama2_70b_cfg = MODEL_CFG(
    HIDDEN_SIZE=8192, 
    INTER_SIZE=28672, 
    HEAD_NUM=64, 
    KV_HEAD_NUM=8,
    VOCAB_SIZE=32000
)
#llama3.1 8b
llama3_8b_cfg = MODEL_CFG(
    HIDDEN_SIZE=4096, 
    INTER_SIZE=14336, 
    HEAD_NUM=32, 
    KV_HEAD_NUM=8, 
    EPS=1e-5,
    VOCAB_SIZE=128256
)

#llama3.3 70b
llama3_70b_cfg = MODEL_CFG(
    HIDDEN_SIZE=8192, 
    INTER_SIZE=28672, 
    HEAD_NUM=64, 
    KV_HEAD_NUM=8, 
    EPS=1e-5,
    VOCAB_SIZE=128256
)

qwen_72b_cfg = MODEL_CFG(
    HIDDEN_SIZE=8192,
    INTER_SIZE=29568,
    HEAD_NUM=64,
    KV_HEAD_NUM=8,
    VOCAB_SIZE=152064,
    MMQKV_BIAS=True,
    TP=2,
    DTYPE=torch.float16,
)

qwen_7b_cfg = MODEL_CFG(
    HIDDEN_SIZE=3584,
    INTER_SIZE=18944,
    HEAD_NUM=28,
    KV_HEAD_NUM=4,
    VOCAB_SIZE=152064,
    MMQKV_BIAS=True,
    TP=2,
    DTYPE=torch.float16,
)


def test_attn(CFG: MODEL_CFG, batch, seqlen, profile_mode):
    # assert seqlen == 1
    net_tpu = SophLlamaAttention(CFG, False, profile_mode)
    net_cpu = LlamaAttention(CFG)
    qkv_tpu = torch.rand(
        (batch, CFG.D * (CFG.HEAD_NUM // CFG.TP + 2 * CFG.KV_HEAD_NUM // CFG.TP)),
        dtype=CFG.DTYPE,
        device=CFG.DEVICE,
    )
    query_tpu, key_tpu, value_tpu = qkv_tpu.split(
        [
            CFG.D * CFG.HEAD_NUM // CFG.TP,
            CFG.D * CFG.KV_HEAD_NUM // CFG.TP,
            CFG.D * CFG.KV_HEAD_NUM // CFG.TP,
        ],
        dim=1,
    )
    query_tpu = query_tpu.view(-1, CFG.HEAD_NUM // CFG.TP, CFG.D)
    kv_view = lambda x: x.view(-1, CFG.KV_HEAD_NUM // CFG.TP, CFG.D)
    key_tpu = kv_view(key_tpu)
    value_tpu = kv_view(value_tpu)
    kv_cache_tpu = (
        torch.rand(
            (CFG.NUM_BLOCKS, CFG.BLOCK_SIZE, CFG.KV_HEAD_NUM // CFG.TP, CFG.D),
            dtype=CFG.DTYPE,
            device=CFG.DEVICE,
        ),
        torch.rand(
            (CFG.NUM_BLOCKS, CFG.BLOCK_SIZE, CFG.KV_HEAD_NUM // CFG.TP, CFG.D),
            dtype=CFG.DTYPE,
            device=CFG.DEVICE,
        ),
    )
    cos_tpu = torch.rand((batch, 1, CFG.D), dtype=CFG.DTYPE, device=CFG.DEVICE)
    sin_tpu = torch.rand((batch, 1, CFG.D), dtype=CFG.DTYPE, device=CFG.DEVICE)
    mask = None
    input_lengths_tpu = torch.tensor(
        [cfg.DECODE_START + seqlen -1] * batch, dtype=torch.int32, device=CFG.DEVICE
    )
    cfg.MAX_BLOCKS= (cfg.DECODE_START + seqlen -1) // cfg.BLOCK_SIZE + 1
    cfg.MAX_SEQLEN= cfg.MAX_BLOCKS * cfg.BLOCK_SIZE
    save_slots_tpu = torch.tensor(
        [[CFG.DECODE_START + seqlen - 2 + b * CFG.MAX_SEQLEN] for b in range(batch)],
        dtype=torch.int32,
        device=CFG.DEVICE,
    )
    fetch_slots_tpu = torch.tensor(
        [
            [
                i
                for i in range(
                    b * CFG.MAX_SEQLEN,
                    b * CFG.MAX_SEQLEN + CFG.DECODE_START + seqlen -1,
                    CFG.BLOCK_SIZE,
                )
            ]
            for b in range(batch)
        ],
        dtype=torch.int32,
        device=CFG.DEVICE,
    )
    block_tables_tpu = torch.tensor(
        [
            [
                i
                for i in range(
                    b * cfg.MAX_BLOCKS,
                    b * cfg.MAX_BLOCKS + (cfg.DECODE_START + seqlen - 1) // cfg.BLOCK_SIZE + 1,
                    1,
                )
            ]
            for b in range(batch)
        ],
        dtype=torch.int32,
        device=CFG.DEVICE,
    )
    slot_size_tpu = block_tables_tpu.shape[1]
    output_tpu = torch.empty_like(query_tpu)

    query_cpu = query_tpu.cpu()
    key_cpu = key_tpu.cpu()
    value_cpu = value_tpu.cpu()
    kcache_cpu = kv_cache_tpu[0].cpu()
    vcache_cpu = kv_cache_tpu[1].cpu()
    cos_cpu = cos_tpu.cpu()
    sin_cpu = sin_tpu.cpu()
    save_slots_cpu = save_slots_tpu.cpu()
    block_tables_cpu = block_tables_tpu.cpu()
    mask_cpu = mask.cpu() if mask is not None else None

    logger.info(
        f"test_parameters:\n{output_tpu.shape}\n{query_tpu.shape=}\n{key_tpu.shape=}\n{value_tpu.shape=}\n\
{kv_cache_tpu[0].shape=}\n{kv_cache_tpu[1].shape=}\n{cos_tpu.shape=}\n{sin_tpu.shape=}\n{mask=}\
{input_lengths_tpu.cpu()=}\n{save_slots_tpu.cpu()=}\n{slot_size_tpu=}\n{block_tables_tpu.cpu()=}"
    )
    net_tpu(
        output_tpu,
        query_tpu,
        key_tpu,
        value_tpu,
        kv_cache_tpu,
        cos_tpu,
        sin_tpu,
        input_lengths_tpu.cpu(),
        save_slots_tpu,
        block_tables_tpu,
        mask,
        slot_size_tpu,
        CFG.DECODE_START + seqlen -1,
        CFG.BLOCK_SIZE,
    )
    output_cpu = net_cpu(
        query_cpu,
        key_cpu,
        value_cpu,
        kcache_cpu,
        vcache_cpu,
        cos_cpu,
        sin_cpu,
        input_lengths_tpu.cpu(),
        mask_cpu,
        save_slots_cpu,
        block_tables_cpu,
        CFG.DECODE_START + seqlen -1,
    )
    return output_tpu, output_cpu


def test_attn_prefill(CFG: MODEL_CFG, batch, seqlen, profile_mode):
    net_tpu = SophLlamaAttention(CFG, True, profile_mode)
    net_cpu = LlamaAttention(CFG, is_prefill=True)
    qkv_tpu = torch.rand(
        (
            batch * seqlen,
            CFG.D * (CFG.HEAD_NUM // CFG.TP + 2 * CFG.KV_HEAD_NUM // CFG.TP),
        ),
        dtype=CFG.DTYPE,
        device=CFG.DEVICE,
    )
    query_tpu, key_tpu, value_tpu = qkv_tpu.split(
        [
            CFG.D * CFG.HEAD_NUM // CFG.TP,
            CFG.D * CFG.KV_HEAD_NUM // CFG.TP,
            CFG.D * CFG.KV_HEAD_NUM // CFG.TP,
        ],
        dim=1,
    )
    query_tpu = query_tpu.view(-1, CFG.HEAD_NUM // CFG.TP, CFG.D)
    kv_view = lambda x: x.view(-1, CFG.KV_HEAD_NUM // CFG.TP, CFG.D)
    key_tpu = kv_view(key_tpu)
    value_tpu = kv_view(value_tpu)
    kv_cache_tpu = (
        torch.rand(
            (CFG.NUM_BLOCKS, CFG.BLOCK_SIZE, CFG.KV_HEAD_NUM // CFG.TP, CFG.D),
            dtype=CFG.DTYPE,
            device=CFG.DEVICE,
        ),
        torch.rand(
            (CFG.NUM_BLOCKS, CFG.BLOCK_SIZE, CFG.KV_HEAD_NUM // CFG.TP, CFG.D),
            dtype=CFG.DTYPE,
            device=CFG.DEVICE,
        ),
    )
    cos_tpu = torch.rand((batch * seqlen, 1, CFG.D), dtype=CFG.DTYPE, device=CFG.DEVICE)
    sin_tpu = torch.rand((batch * seqlen, 1, CFG.D), dtype=CFG.DTYPE, device=CFG.DEVICE)
    mask = torch.empty((seqlen, seqlen), dtype=CFG.DTYPE, device=CFG.DEVICE)
    input_lengths_tpu = torch.tensor(
        [seqlen] * batch, dtype=torch.int32, device=CFG.DEVICE
    )
    cfg.MAX_BLOCKS = (cfg.DECODE_START + seqlen -1) // cfg.BLOCK_SIZE + 1
    cfg.MAX_SEQLEN= cfg.MAX_BLOCKS * cfg.BLOCK_SIZE
    block_tables_tpu = torch.tensor(
        [
            [
                i
                for i in range(
                    b * CFG.MAX_BLOCKS, (b+1) * CFG.MAX_BLOCKS, 1
                )
            ]
            for b in range(batch)
        ],
        dtype=torch.int32,
        device=CFG.DEVICE,
    )
    slot_size_tpu = CFG.MAX_SEQLEN // CFG.BLOCK_SIZE
    output_tpu = torch.empty_like(query_tpu)

    query_cpu = query_tpu.cpu()
    key_cpu = key_tpu.cpu()
    value_cpu = value_tpu.cpu()
    kcache_cpu = kv_cache_tpu[0].cpu()
    vcache_cpu = kv_cache_tpu[1].cpu()
    cos_cpu = cos_tpu.cpu()
    sin_cpu = sin_tpu.cpu()
    block_tables_cpu = block_tables_tpu.cpu()
    mask_cpu = None

    logger.info(
        f"test_parameters:\n{output_tpu.shape}\n{query_tpu.shape=}\n{key_tpu.shape=}\n{value_tpu.shape=}\n\
{kv_cache_tpu[0].shape=}\n{kv_cache_tpu[1].shape=}\n{cos_tpu.shape=}\n{sin_tpu.shape=}\n{mask.shape=}\
{input_lengths_tpu.cpu()=}\n{block_tables_tpu.cpu()=}\n{slot_size_tpu=}"
    )
    net_tpu(
        output_tpu,
        query_tpu,
        key_tpu,
        value_tpu,
        kv_cache_tpu,
        cos_tpu,
        sin_tpu,
        input_lengths_tpu.cpu(),
        block_tables_tpu,
        block_tables_tpu,
        mask,
        slot_size_tpu,
        seqlen,
        CFG.BLOCK_SIZE,
    )
    output_cpu = net_cpu(
        query_cpu,
        key_cpu,
        value_cpu,
        kcache_cpu,
        vcache_cpu,
        cos_cpu,
        sin_cpu,
        input_lengths_tpu.cpu(),
        mask_cpu,
        block_tables_cpu,
        block_tables_cpu,
        seqlen,
    )
    return output_tpu, output_cpu


def test_base(model_class, cfg: MODEL_CFG, in_tensors, profile_mode):
    if in_tensors[0].device.type == "tpu":
        net = model_class(cfg, profile_mode=profile_mode)
    else:
        net = model_class(cfg)
    return net(*in_tensors)


def test_add(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):
    hidden_states_cpu = torch.rand(
        (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE
    )
    output_cpu = torch.rand(
        (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE
    )
    hidden_states_tpu = hidden_states_cpu.clone().to(cfg.DEVICE)
    output_tpu = output_cpu.clone().to(cfg.DEVICE)
    # print(f"hidden_states_tpu:{hidden_states_tpu.cpu()}")
    # print(f"output_tpu:{output_tpu.cpu()}")
    # print(f"hidden_states_cpu:{hidden_states_cpu}")
    # print(f"output_cpu:{output_cpu}")
    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(hidden_states_tpu, output_tpu),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, output_cpu),
        profile_mode=profile_mode,
    )


def test_rmsnorm(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):
    hidden_states_tpu = torch.rand(
        (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
    )
    weight_tpu = torch.rand((cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE)
    output_tpu = torch.empty_like(hidden_states_tpu)

    hidden_states_cpu = hidden_states_tpu.clone().cpu()
    weight_cpu = weight_tpu.clone().cpu()
    output_cpu = torch.empty_like(hidden_states_cpu)

    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(hidden_states_tpu, weight_tpu, output_tpu),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, weight_cpu, output_cpu),
        profile_mode=profile_mode,
    )


def test_mmqkv(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):
    hidden_states_tpu = torch.rand(
        (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
    )
    weight_tpu = (
        torch.rand(
            (
                cfg.HIDDEN_SIZE,
                cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
            ),
            dtype=cfg.DTYPE,
        )
        .T.contiguous()
        .to(cfg.DEVICE)
    )
    bias_tpu = (
        torch.rand(
            (cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP)),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        if cfg.MMQKV_BIAS
        else None
    )

    hidden_states_cpu = hidden_states_tpu.clone().cpu()
    weight_cpu = weight_tpu.clone().cpu().T

    bias_cpu = bias_tpu.clone().cpu() if cfg.MMQKV_BIAS else None

    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(hidden_states_tpu, weight_tpu, bias_tpu),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, weight_cpu, bias_cpu),
        profile_mode=profile_mode,
    )


def test_mmqkv_w4a16(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):
    group_size = 128
    hidden_states_cpu = torch.randn(
        (batch * seqlen, cfg.HIDDEN_SIZE),
        dtype=cfg.DTYPE,
    )
    hidden_states_tpu = hidden_states_cpu.clone().to(cfg.DEVICE)

    qzeros_tpu = torch.full(
        [
            cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
            cfg.D * cfg.HEAD_NUM // group_size,
        ],
        3,
        dtype=torch.uint8,
    )
    qzeros_tpu = torch.bitwise_or(
        qzeros_tpu.transpose(-1, -2)[::2], qzeros_tpu.transpose(-1, -2)[1::2] << 4
    ).transpose(-1, -2)

    scales_ori = torch.full(
        [
            cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
            cfg.D * cfg.HEAD_NUM // group_size,
        ],
        0.1,
        dtype=cfg.DTYPE,
    )
    scales_tpu = torch.cat(
        (scales_ori.view(dtype=torch.uint8), qzeros_tpu), axis=-1
    ).contiguous()

    bias_tpu = (
        torch.randn(
            (cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        if cfg.MMQKV_BIAS
        else None
    )
    bias_cpu = bias_tpu.clone().cpu()if cfg.MMQKV_BIAS else None

    qweight_int8_tensor = torch.randint(
        4,
        15,
        (
            cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
            cfg.D * cfg.HEAD_NUM,
        ),
        dtype=torch.uint8,
    )
    qweight_cpu_qt = torch.bitwise_or(
        qweight_int8_tensor.transpose(-1, -2)[::2],
        qweight_int8_tensor.transpose(-1, -2)[1::2] << 4,
    ).transpose(-1, -2)
    qweight_cpu_deq = (qweight_int8_tensor - 3) * 0.1
    qweight_cpu_deq = qweight_cpu_deq.T

    print(f"qweight_cpu_deq{qweight_cpu_deq}")
    print(f"hidden_states_cpu{hidden_states_cpu}")

    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(
            hidden_states_tpu,
            qweight_cpu_qt.to(cfg.DEVICE),
            bias_tpu,
            qzeros_tpu.to(cfg.DEVICE),
            scales_tpu.to(cfg.DEVICE),
        ),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, qweight_cpu_deq, bias_cpu),
        profile_mode=profile_mode,
    )


def test_mlp(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):
    hidden_states_tpu = torch.rand(
        (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
    )
    w0_tpu = -1 + 2 * torch.rand(
        (cfg.HIDDEN_SIZE, cfg.INTER_SIZE // cfg.TP),
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )
    w1_tpu = -1 + 2 * torch.rand(
        (cfg.HIDDEN_SIZE, cfg.INTER_SIZE // cfg.TP),
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )
    w2_tpu = -1 + 2 * torch.rand(
        (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE),
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )
    output_tpu = torch.empty_like(hidden_states_tpu)

    hidden_states_cpu = hidden_states_tpu.clone().cpu()
    w0_cpu = w0_tpu.clone().cpu()
    w1_cpu = w1_tpu.clone().cpu()
    w2_cpu = w2_tpu.clone().cpu()
    output_cpu = torch.empty_like(hidden_states_cpu)

    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(hidden_states_tpu, w0_tpu, w1_tpu, w2_tpu, output_tpu),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, w0_cpu, w1_cpu, w2_cpu, output_cpu),
        profile_mode=profile_mode,
    )


def dequant(q_group_size, weight_bits, qweight, qzeros, qscale):
    i = 0
    col_ = 0
    print(f"{qweight.shape=}")
    print(f"{qzeros.shape=}")
    weight_split = torch.empty(
        (qweight.shape[0], qweight.shape[1] * 8 // weight_bits), dtype=torch.uint8
    )
    while col_ < qweight.shape[1]:
        for j in range(i, i + (8 // weight_bits)):
            weight_split[:, j] = (qweight[:, col_] >> weight_bits * (j - i)) & 0xF
        i += 8 // weight_bits
        col_ += 1
    # split z
    i = 0
    col_ = 0
    zeros_split = torch.empty(
        (qzeros.shape[0], qzeros.shape[1] * 8 // weight_bits), dtype=torch.uint8
    )
    while col_ < qzeros.shape[1]:
        for j in range(i, i + (8 // weight_bits)):
            zeros_split[:, j] = (qzeros[:, col_] >> weight_bits * (j - i)) & 0xF
        i += 8 // weight_bits
        col_ += 1
    # w = w - z
    zeros = torch.empty(
        (zeros_split.shape[0], zeros_split.shape[1] * q_group_size), dtype=torch.int8
    )
    for i in range(zeros_split.shape[1]):
        zeros[:, i * q_group_size : (i + 1) * q_group_size] = zeros_split[:, i : i + 1]
    # if zeros.shape[0]>=weight_split.shape[0]:
    #     zeros = zeros[: weight_split.shape[0], : weight_split.shape[1]]
    # else:
    #     zeros=torch.expand(weight_split.shape[0],weight_split.shape[1])
    zeros = zeros[: weight_split.shape[0], : weight_split.shape[1]]   
    print(f"{weight_split.shape=}")
    print(f"{zeros.shape=}")
    dequant_weight = weight_split - zeros
    # dequant_weigth = w * s
    scale = torch.empty(
        (qscale.shape[0], qscale.shape[1] * q_group_size), dtype=torch.float16
    )
    print(f"{qscale.shape=}")
    for i in range(qscale.shape[1]):
        scale[:, i * q_group_size : (i + 1) * q_group_size] = qscale[:, i : i + 1]
    print(f"{qscale.shape=}")
    scale = scale[: weight_split.shape[0], : weight_split.shape[1]]
    print(scale)
    dequant_weight = dequant_weight * scale
    return dequant_weight


def test_mlp_w4a16(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):  
    group_size = 128
    zp_value = 119
    hidden_states_tpu = torch.rand(
        (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
    ).contiguous()
    w0_cpu = (
        torch.randint(
            0,
            255,
            (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // 2),
        )
        .contiguous()
        .to(torch.uint8)
    )
    w1_cpu = (
        torch.randint(
            0,
            255,
            (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // 2),
        )
        .contiguous()
        .to(torch.uint8)
    )
    w2_cpu = (
        torch.randint(
            0,
            255,
            (cfg.HIDDEN_SIZE, int(math.ceil(cfg.INTER_SIZE / 2/ cfg.TP))),
        )
        .contiguous()
        .to(torch.uint8)
    )
    output_tpu = torch.empty_like(hidden_states_tpu).contiguous()

    
    up_qzeros_cpu = torch.full(
        (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size // 2),
        zp_value,
        dtype=torch.uint8,
    ).contiguous()
    up_scales_cpu = torch.rand(
        (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size),
        dtype=cfg.DTYPE,
    ).contiguous()*(0.005-0.002)+0.002
    gate_qzeros_cpu = torch.full(
        (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size // 2),
        zp_value,
        dtype=torch.uint8,
    ).contiguous()
    gate_scales_cpu = torch.rand(
        (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size),

        dtype=cfg.DTYPE,
    ).contiguous()*(0.005-0.002)+0.002        
    down_qzeros_cpu = torch.full(
        (cfg.HIDDEN_SIZE, int(math.ceil(cfg.INTER_SIZE / group_size / 2 / cfg.TP))),
        zp_value,
        dtype=torch.uint8,
    ).contiguous()
    down_scales_cpu = torch.rand(
        (cfg.HIDDEN_SIZE, int(math.ceil(cfg.INTER_SIZE / group_size / 2 / cfg.TP)*2)),
        dtype=cfg.DTYPE,
    ).contiguous()*(0.005-0.002)+0.002

    up_qzeros_tpu = up_qzeros_cpu.clone().contiguous().to(cfg.DEVICE)
    up_scales_tpu = up_scales_cpu.clone().contiguous().to(cfg.DEVICE)
    gate_qzeros_tpu = gate_qzeros_cpu.clone().contiguous().to(cfg.DEVICE)
    gate_scales_tpu = gate_scales_cpu.clone().contiguous().to(cfg.DEVICE)
    down_scales_tpu = down_scales_cpu.clone().T.contiguous().to(cfg.DEVICE)
    down_qzeros_tpu = down_qzeros_cpu.clone().T.contiguous().to(cfg.DEVICE)
    print(f"{up_qzeros_tpu.shape=}")
    print(f"{up_scales_tpu.shape=}")
    print(f"{gate_qzeros_tpu.shape=}")
    print(f"{gate_scales_tpu.shape=}")
    print(f"{down_qzeros_tpu.shape=}")
    print(f"{down_scales_tpu.shape=}")
    hidden_states_cpu = hidden_states_tpu.clone().cpu()
    w0_tpu = w1_cpu.clone().contiguous().to(cfg.DEVICE)
    w0_cpu_deq = dequant(128, 4, w0_cpu, up_qzeros_cpu, up_scales_cpu).T
    w1_tpu = w1_cpu.clone().contiguous().to(cfg.DEVICE)
    w1_cpu_deq = dequant(128, 4, w1_cpu, gate_qzeros_cpu, gate_scales_cpu).T
    w2_tpu = w2_cpu.clone().t().view(-1,group_size,cfg.HIDDEN_SIZE).permute(0,2,1).contiguous().view(-1,group_size*cfg.HIDDEN_SIZE).contiguous().to(cfg.DEVICE)
    w2_cpu_deq = dequant(128, 4, w2_cpu, down_qzeros_cpu, down_scales_cpu).T
    output_cpu = torch.empty_like(hidden_states_cpu).contiguous()
    print(f"w1_cpu_deq={w1_cpu_deq}")
    print(f"w0_cpu_deq={w0_cpu_deq}")
    print(f"w2_cpu_deq={w2_cpu_deq}")
    print(f"{up_qzeros_tpu.shape=}")
    print(f"{up_scales_tpu.shape=}")
    print(f"{w0_tpu.shape=}")
    print(f"{gate_qzeros_tpu.shape=}")
    print(f"{gate_scales_tpu.shape=}")
    print(f"{w1_tpu.shape=}")
    print(f"{down_qzeros_tpu.shape=}")
    print(f"{down_scales_tpu.shape=}")
    print(f"{w2_tpu.shape=}")
    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(
            hidden_states_tpu,
            w0_tpu,
            up_qzeros_tpu,
            up_scales_tpu,
            w1_tpu,
            gate_qzeros_tpu,
            gate_scales_tpu,
            w2_tpu,
            down_qzeros_tpu,
            down_scales_tpu,
            output_tpu,
        ),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, w0_cpu_deq, w1_cpu_deq, w2_cpu_deq, output_cpu),
        profile_mode=profile_mode,
    )


def test_attn_fc(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):
    hidden_states_tpu = torch.rand(
        (batch * seqlen, cfg.HIDDEN_SIZE // cfg.TP), dtype=cfg.DTYPE, device=cfg.DEVICE
    )
    weight_tpu = torch.rand(
        (cfg.HIDDEN_SIZE, cfg.D * cfg.HEAD_NUM // cfg.TP),
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )

    print(f"{weight_tpu.shape=}\n")
    print(f"{hidden_states_tpu.shape=}\n")

    hidden_states_cpu = hidden_states_tpu.clone().cpu()
    weight_cpu = weight_tpu.clone().cpu().T

    print(f"{weight_tpu.shape=}\n")
    print(f"{hidden_states_tpu.shape=}\n")
    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(hidden_states_tpu, weight_tpu),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, weight_cpu),
        profile_mode=profile_mode,
    )


def test_attn_fc_w4a16(
    model_class_tpu, model_class_cpu, cfg: MODEL_CFG, batch, seqlen, profile_mode,
):
    group_size = 128
    hidden_states_cpu = torch.randn(
        (batch * seqlen, cfg.HIDDEN_SIZE // cfg.TP),
        dtype=cfg.DTYPE,
    )
    hidden_states_tpu = hidden_states_cpu.clone().to(cfg.DEVICE)
    print(f"{hidden_states_tpu.shape=}")

    qzeros_tpu = torch.full(
        [cfg.HIDDEN_SIZE, cfg.D * cfg.HEAD_NUM // cfg.TP // group_size],
        3,
        dtype=torch.uint8,
    )

    if qzeros_tpu.size(1) % 2 != 0:
        padding = torch.full([cfg.HIDDEN_SIZE, 1], 3, dtype=torch.uint8)
        qzeros_tpu = torch.cat([qzeros_tpu, padding], dim=1)
        print(f"After padding, qzeros_tpu shape: {qzeros_tpu.shape}")

    qzeros_tpu_transposed = qzeros_tpu.transpose(-1, -2)
    qzeros_tpu = torch.bitwise_or(
        qzeros_tpu_transposed[::2], qzeros_tpu_transposed[1::2] << 4
    ).transpose(-1, -2)

    scales_ori = torch.full(
        [cfg.HIDDEN_SIZE, cfg.D * cfg.HEAD_NUM // cfg.TP // group_size],
        0.1,
        dtype=cfg.DTYPE,
    )
    scales_tpu = torch.cat(
        (scales_ori.view(dtype=torch.uint8), qzeros_tpu), axis=-1
    ).contiguous()

    bias_tpu = None

    output_tpu = torch.empty(
        (batch * seqlen, cfg.HIDDEN_SIZE),
        dtype=cfg.DTYPE,
        device=cfg.DEVICE,
    )

    qweight_int8_tensor = torch.randint(
        4, 15, (cfg.HIDDEN_SIZE, cfg.D * cfg.HEAD_NUM // cfg.TP), dtype=torch.uint8
    )
    qweight_cpu_qt = torch.bitwise_or(
        qweight_int8_tensor.transpose(-1, -2)[::2],
        qweight_int8_tensor.transpose(-1, -2)[1::2] << 4,
    ).transpose(-1, -2)
    # print(f"{qweight_cpu_deq.shape=}")
    qweight_cpu_deq = (qweight_int8_tensor - 3) * 0.1
    qweight_cpu_deq = qweight_cpu_deq.T

    # print(f"qweight_cpu_deq{qweight_cpu_deq}")
    print(f"{qweight_cpu_deq.shape=}")
    print(f"hidden_states_cpu{hidden_states_cpu}")
    print(f"qweight_cpu_deq{qweight_cpu_deq}")
    return test_base(
        model_class=model_class_tpu,
        cfg=cfg,
        in_tensors=(
            hidden_states_tpu,
            qweight_cpu_qt.to(cfg.DEVICE),
            qzeros_tpu.to(cfg.DEVICE),
            scales_tpu.to(cfg.DEVICE),
            output_tpu,
        ),
        profile_mode=profile_mode,
    ), test_base(
        model_class=model_class_cpu,
        cfg=cfg,
        in_tensors=(hidden_states_cpu, qweight_cpu_deq),
        profile_mode=profile_mode,
    )


def check_add(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, is_prefill, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_add(
            SophLlamaAdd,
            LlamaAdd,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False:
            print(f"[Failed] llama_add compare failed!")
            sys.exit(255)
        print(f"[Success] llama_add compare pass!")
    else:
        if is_prefill:
            path = "prefill_add"
        else:
            path = "decode_add"
        if is_PerfAI:        
            generate_run_perfai_script(path)
        if profile_mode != None:
            generate_run_profile_script(path)

        hidden_states_tpu = torch.rand(
            (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        output_tpu = torch.empty_like(hidden_states_tpu)
        return test_base(
            SophLlamaAdd,
            cfg=cfg,
            in_tensors=(hidden_states_tpu, output_tpu),
            profile_mode=profile_mode,
        )

def check_rmsnorm(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, is_prefill, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_rmsnorm(
            SophLlamaRMSNorm,
            LlamaRMSNorm,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False:
            print(f"[Failed] llama_rmsnorm compare failed!")
            sys.exit(255)
        print(f"[Success] llama_rmsnorm compare pass!")
    else:
        if is_prefill:
            path = "prefill_rmsnorm"
        else:
            path = "decode_rmsnorm"
        if is_PerfAI:        
            generate_run_perfai_script(path)
        if profile_mode != None:
            generate_run_profile_script(path)
        hidden_states_tpu = torch.rand(
            (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        weight_tpu = torch.rand(
            (cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        output_tpu = torch.empty_like(hidden_states_tpu)

        return test_base(
            SophLlamaRMSNorm,
            cfg=cfg,
            in_tensors=(hidden_states_tpu, weight_tpu, output_tpu),
            profile_mode=profile_mode,
        )


def check_mmqkv(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, is_prefill, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_mmqkv(
            SophLlamaMMqkv,
            LlamaMMqkv,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False > 0.01:
            print(f"[Failed] llama_mmqkv compare failed!")
            sys.exit(255)
        print(f"[Success] llama_mmkqv compare pass!")
    else:
        if is_prefill:
            path = "prefill_mmqkv"
        else:
            path = "decode_mmqkv"
        if is_PerfAI:        
            generate_run_perfai_script(path)
        if profile_mode != None:
            generate_run_profile_script(path)
        hidden_states_tpu = torch.rand(
            (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        weight_tpu = (
            torch.rand(
                (
                    cfg.HIDDEN_SIZE,
                    cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
                ),
                dtype=cfg.DTYPE,
            )
            .T.contiguous()
            .to(cfg.DEVICE)
        )
        bias_tpu = (
            torch.rand(
                (cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP)),
                dtype=cfg.DTYPE,
                device=cfg.DEVICE,
            )
            if cfg.MMQKV_BIAS
            else None
        )

        return test_base(
            SophLlamaMMqkv,
            cfg=cfg,
            in_tensors=(hidden_states_tpu, weight_tpu, bias_tpu),
            profile_mode=profile_mode,
        )


def check_mmqkv_w4a16(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_mmqkv_w4a16(
            SophLlamaMMqkvW4a16,
            LlamaMMqkv,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print(f"{out_cpu.dtype=}, {out_tpu.dtype=}")
        print(f"{out_cpu.shape=}, {out_tpu.shape=}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False :
            print(f"[Failed] llama_mmqkv_w4a16 compare failed!")
            sys.exit(255)
        print(f"[Success] llama_mmqkv_w4a16 compare pass!")
    else:
        if is_PerfAI:
            path = "w4a16_mmqkv"
            generate_run_perfai_script(path)
        if profile_mode != None:
            path = "w4a16_mmqkv"
            generate_run_profile_script(path)
        group_size = 128
        hidden_states_tpu = torch.rand(
            (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        qweight_tpu = (
            torch.empty(
                (
                    cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
                    cfg.HIDDEN_SIZE // 2,
                ),
                dtype=torch.uint8,
            )
            .contiguous()
            .to(cfg.DEVICE)
        )
        qzeros_tpu = torch.empty(
            (
                cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
                cfg.HIDDEN_SIZE // group_size // 2,
            ),
            device=cfg.DEVICE,
            dtype=torch.uint8,
        )
        scales_tpu = torch.empty(
            (
                cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
                cfg.HIDDEN_SIZE // group_size * 2 + cfg.HIDDEN_SIZE // group_size // 2,
            ),
            device=cfg.DEVICE,
            dtype=torch.uint8,
        )
        bias_tpu = (
            torch.empty(
                (cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP)),
                dtype=cfg.DTYPE,
                device=cfg.DEVICE,
            )
            if cfg.MMQKV_BIAS
            else None
        )
        return test_base(
            SophLlamaMMqkvW4a16,
            cfg=cfg,
            in_tensors=(
                hidden_states_tpu,
                qweight_tpu,
                bias_tpu,
                qzeros_tpu,
                scales_tpu,
            ),
            profile_mode=profile_mode,
        )


def check_mlp(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, is_prefill, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_mlp(
            SophLlamaMlp,
            LlamaMlp,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False :
            print(f"[Failed] llama_mlp compare failed!")
            sys.exit(255)
        print(f"[Success] llama_mlp compare pass!")
    else:
        if is_prefill:
            path = "prefill_mlp"
        else:
            path = "decode_mlp"
        if is_PerfAI:        
            generate_run_perfai_script(path)
        if profile_mode != None:
            generate_run_profile_script(path)

        hidden_states_tpu = torch.rand(
            (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        w0_tpu = -1 + 2 * torch.rand(
            (cfg.HIDDEN_SIZE, cfg.INTER_SIZE // cfg.TP),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        w1_tpu = -1 + 2 * torch.rand(
            (cfg.HIDDEN_SIZE, cfg.INTER_SIZE // cfg.TP),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        w2_tpu = -1 + 2 * torch.rand(
            (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        output_tpu = torch.empty_like(hidden_states_tpu)

        return test_base(
            model_class=SophLlamaMlp,
            cfg=cfg,
            in_tensors=(hidden_states_tpu, w0_tpu, w1_tpu, w2_tpu, output_tpu),
            profile_mode=profile_mode,
        )


def check_mlp_w4a16(cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, profile_mode):
    if is_test:
        out_tpu, out_cpu = test_mlp_w4a16(
            SophLlamaMlpW4a16,
            LlamaMlpW4a16,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False:
            print(f"[Failed] llama_mlp_w4a16 compare failed!")
            sys.exit(255)
        print(f"[Success] llama_mlp_w4a16 compare pass!")
    else:
        if is_PerfAI:
            path = "w4a16_mlp"
            generate_run_perfai_script(path)
        if profile_mode != None:
            path = "w4a16_mlp"
            generate_run_profile_script(path)
        zp_value = 119
        group_size = 128
        hidden_states_tpu = torch.rand(
            (batch * seqlen, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        w0_tpu = (
            torch.randint(
                0,
                255,
                (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // 2),
                device=cfg.DEVICE,
            )
            .contiguous()
            .to(torch.uint8)
        )
        w1_tpu = (
            torch.randint(
                0,
                255,
                (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // 2),
                device=cfg.DEVICE,
            )
            .contiguous()
            .to(torch.uint8)
        )
        w2_tpu = (
            torch.randint(
                0,
                255,
                (cfg.INTER_SIZE // group_size // 2 // cfg.TP, cfg.HIDDEN_SIZE * 128),
                device=cfg.DEVICE,
            )
            .contiguous()
            .to(torch.uint8)
        )

        output_tpu = torch.empty_like(hidden_states_tpu)

        up_qzeros_cpu = torch.full(
            (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size // 2),
            zp_value,
            dtype=torch.uint8,
        ).contiguous()
        up_scales_cpu = torch.full(
            (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size),
            0.1,
            dtype=cfg.DTYPE,
        ).contiguous()
        gate_qzeros_cpu = torch.full(
            (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size // 2),
            zp_value,
            dtype=torch.uint8,
        ).contiguous()
        gate_scales_cpu = torch.full(
            (cfg.INTER_SIZE // cfg.TP, cfg.HIDDEN_SIZE // group_size),
            0.1,
            dtype=cfg.DTYPE,
        ).contiguous()
        down_qzeros_cpu = torch.full(
            (cfg.HIDDEN_SIZE, cfg.INTER_SIZE // group_size // 2 // cfg.TP),
            zp_value,
            dtype=torch.uint8,
        ).contiguous()
        down_scales_cpu = torch.full(
            (cfg.HIDDEN_SIZE, cfg.INTER_SIZE // group_size // cfg.TP),
            0.1,
            dtype=cfg.DTYPE,
        ).contiguous()

        up_qzeros_tpu = up_qzeros_cpu.to(cfg.DEVICE)
        up_scales_tpu = up_scales_cpu.to(cfg.DEVICE)
        gate_qzeros_tpu = gate_qzeros_cpu.to(cfg.DEVICE)
        gate_scales_tpu = gate_scales_cpu.to(cfg.DEVICE)
        down_qzeros_tpu = down_qzeros_cpu.T.to(cfg.DEVICE)
        down_scales_tpu = down_scales_cpu.T.to(cfg.DEVICE)

        return test_base(
            model_class=SophLlamaMlpW4a16,
            cfg=cfg,
            in_tensors=(
                hidden_states_tpu,
                w0_tpu,
                up_qzeros_tpu,
                up_scales_tpu,
                w1_tpu,
                gate_qzeros_tpu,
                gate_scales_tpu,
                w2_tpu,
                down_qzeros_tpu,
                down_scales_tpu,
                output_tpu,
            ),
            profile_mode=profile_mode,
        )


def check_attn_fc(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, is_prefill, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_attn_fc(
            SophLlamaAttentionFC,
            LlamaAttentionFC,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print(f"{out_cpu.shape=}")
        print(f"{out_tpu.shape=}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False:
            print(f"[Failed] llama_attn_fc compare failed!")
            sys.exit(255)
        print(f"[Success] llama_attn_fc compare pass!")
    else:
        if is_prefill:
            path = "prefill_attn_fc"
        else:
            path = "decode_attn_fc"
        if is_PerfAI:        
            generate_run_perfai_script(path)
        if profile_mode != None:
            generate_run_profile_script(path)
        hidden_states_tpu = torch.rand(
            (batch * seqlen, cfg.HIDDEN_SIZE // cfg.TP),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        weight_tpu = (
            torch.rand(
                (cfg.D * cfg.HEAD_NUM // cfg.TP, cfg.HIDDEN_SIZE), dtype=cfg.DTYPE
            )
            .T.contiguous()
            .to(cfg.DEVICE)
        )

        return test_base(
            SophLlamaAttentionFC,
            cfg=cfg,
            in_tensors=(hidden_states_tpu, weight_tpu),
            profile_mode=profile_mode,
        )


def check_attn_fc_w4a16(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_attn_fc_w4a16(
            SophLlamaAttentionFcW4a16,
            LlamaAttentionFcW4a16,
            cfg=cfg,
            batch=batch,
            seqlen=seqlen,
            profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float(), out_tpu.detach().float()
        )
        if status == False:
            print(f"[Failed] llama_attn_fc_w4a16 compare failed!")
            sys.exit(255)
        print(f"[Success] llama_attn_fc_w4a16 compare pass!")
    else:
        if is_PerfAI:
            path = "w4a16_attn_fc"
            generate_run_perfai_script(path)
        if profile_mode != None:
            path = "w4a16_attn_fc"
            generate_run_profile_script(path)
        group_size = 128
        hidden_states_cpu = torch.randn(
            (batch * seqlen, cfg.HIDDEN_SIZE // cfg.TP),
            dtype=cfg.DTYPE,
        )
        hidden_states_tpu = hidden_states_cpu.clone().to(cfg.DEVICE)

        qzeros_tpu = torch.full(
            [cfg.HIDDEN_SIZE, cfg.D * cfg.HEAD_NUM // cfg.TP // group_size],
            3,
            dtype=torch.uint8,
        )

        if qzeros_tpu.size(1) % 2 != 0:
            padding = torch.full([cfg.HIDDEN_SIZE, 1], 3, dtype=torch.uint8)
            qzeros_tpu = torch.cat([qzeros_tpu, padding], dim=1)
            print(f"After padding, qzeros_tpu shape: {qzeros_tpu.shape}")

        qzeros_tpu_transposed = qzeros_tpu.transpose(-1, -2)
        qzeros_tpu = torch.bitwise_or(
            qzeros_tpu_transposed[::2], qzeros_tpu_transposed[1::2] << 4
        ).transpose(-1, -2)

        scales_ori = torch.full(
            [cfg.HIDDEN_SIZE, cfg.D * cfg.HEAD_NUM // cfg.TP // group_size],
            0.1,
            dtype=cfg.DTYPE,
        )
        scales_tpu = torch.cat(
            (scales_ori.view(dtype=torch.uint8), qzeros_tpu), axis=-1
        ).contiguous()

        bias_tpu = None

        output_tpu = torch.empty(
            (batch * seqlen, cfg.HIDDEN_SIZE),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )

        qweight_int8_tensor = torch.randint(
            4, 15, (cfg.HIDDEN_SIZE, cfg.D * cfg.HEAD_NUM // cfg.TP), dtype=torch.uint8
        )
        qweight_cpu_qt = torch.bitwise_or(
            qweight_int8_tensor.transpose(-1, -2)[::2],
            qweight_int8_tensor.transpose(-1, -2)[1::2] << 4,
        ).transpose(-1, -2)

        return test_base(
            model_class=SophLlamaAttentionFcW4a16,
            cfg=cfg,
            in_tensors=(
                hidden_states_tpu,
                qweight_cpu_qt.to(cfg.DEVICE),
                qzeros_tpu.to(cfg.DEVICE),
                scales_tpu.to(cfg.DEVICE),
                output_tpu,
            ),
            profile_mode=profile_mode,
        )


def check_attn_prefill(
    cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, profile_mode
):
    if is_test:
        out_tpu, out_cpu = test_attn_prefill(
            cfg, batch=batch, seqlen=seqlen, profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print(f"{out_cpu.shape=}")
        print(f"{out_tpu.shape=}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float().reshape(-1), out_tpu.detach().float().reshape(-1)
        )
        if status == False:
            print(f"[Failed] llama_attn_prefill compare failed!")
            sys.exit(255)
        print(f"[Success] llama_attn_prefill compare pass!")
    else:
        if is_PerfAI:
            path = "prefill_attn"
            generate_run_perfai_script(path)
        if profile_mode != None:
            path = "prefill_attn"
            generate_run_profile_script(path)
        net_tpu = SophLlamaAttention(cfg, is_prefill=True, profile_mode=profile_mode,)
        qkv_tpu = torch.rand(
            (
                batch * seqlen,
                cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP),
            ),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        query_tpu, key_tpu, value_tpu = qkv_tpu.split(
            [
                cfg.D * cfg.HEAD_NUM // cfg.TP,
                cfg.D * cfg.KV_HEAD_NUM // cfg.TP,
                cfg.D * cfg.KV_HEAD_NUM // cfg.TP,
            ],
            dim=1,
        )
        query_tpu = query_tpu.view(-1, cfg.HEAD_NUM // cfg.TP, cfg.D)
        kv_view = lambda x: x.view(-1, cfg.KV_HEAD_NUM // cfg.TP, cfg.D)
        key_tpu = kv_view(key_tpu)
        value_tpu = kv_view(value_tpu)
        kv_cache_tpu = (
            torch.rand(
                (cfg.NUM_BLOCKS, cfg.BLOCK_SIZE, cfg.KV_HEAD_NUM // cfg.TP, cfg.D),
                dtype=cfg.DTYPE,
                device=cfg.DEVICE,
            ),
            torch.rand(
                (cfg.NUM_BLOCKS, cfg.BLOCK_SIZE, cfg.KV_HEAD_NUM // cfg.TP, cfg.D),
                dtype=cfg.DTYPE,
                device=cfg.DEVICE,
            ),
        )
        cos_tpu = torch.rand(
            (batch * seqlen, 1, cfg.D), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        sin_tpu = torch.rand(
            (batch * seqlen, 1, cfg.D), dtype=cfg.DTYPE, device=cfg.DEVICE
        )
        mask = torch.empty((seqlen, seqlen), dtype=cfg.DTYPE, device=cfg.DEVICE)
        input_lengths_tpu = torch.tensor(
            [seqlen] * batch, dtype=torch.int32, device=cfg.DEVICE
        )
        cfg.MAX_SEQLEN= ((cfg.DECODE_START + seqlen -1) // cfg.BLOCK_SIZE + 1) * cfg.BLOCK_SIZE
        save_slots_tpu = torch.tensor(
            [
                [
                    i
                    for i in range(
                        b * cfg.MAX_SEQLEN, (b+1) * cfg.MAX_SEQLEN, cfg.BLOCK_SIZE
                    )
                ]
                for b in range(batch)
            ],
            dtype=torch.int32,
            device=cfg.DEVICE,
        )
        block_tables_tpu = None
        slot_size_tpu = cfg.MAX_SEQLEN // cfg.BLOCK_SIZE
        output_tpu = torch.empty_like(query_tpu)

        logger.info(
            f"test_parameters:\n{output_tpu.shape}\n{query_tpu.shape=}\n{key_tpu.shape=}\n{value_tpu.shape=}\n\
    {kv_cache_tpu[0].shape=}\n{kv_cache_tpu[1].shape=}\n{cos_tpu.shape=}\n{sin_tpu.shape=}\n{mask.shape=}\
    {input_lengths_tpu.cpu()=}\n{save_slots_tpu.cpu()=}\n{slot_size_tpu=}\n{block_tables_tpu=}"
        )
        net_tpu(
            output_tpu,
            query_tpu,
            key_tpu,
            value_tpu,
            kv_cache_tpu,
            cos_tpu,
            sin_tpu,
            input_lengths_tpu.cpu(),
            save_slots_tpu,
            block_tables_tpu,
            mask,
            slot_size_tpu,
            seqlen,
            cfg.BLOCK_SIZE,
        )


def check_attn(cfg: MODEL_CFG, batch, seqlen, is_test, is_PerfAI, profile_mode):
    if is_test:
        out_tpu, out_cpu = test_attn(
            cfg, batch=batch, seqlen=seqlen, profile_mode=profile_mode,
        )
        out_tpu = out_tpu.cpu()

        print(f"outcpu{out_cpu}")
        print(f"outtpu{out_tpu}")
        print(f"{out_cpu.shape=}")
        print(f"{out_tpu.shape=}")
        print("out_cpu has NaNs:", torch.isnan(out_cpu).any())
        print("out_tpu has NaNs:", torch.isnan(out_tpu).any())
        comparator = TensorComparator()
        status = comparator.cmp_result(
            out_cpu.detach().float().reshape(-1), out_tpu.detach().float().reshape(-1)
        )
        if status == False:
            print(f"[Failed] llama_attn_decode compare failed!")
            sys.exit(255)
        print(f"[Success] llama_attn_decode compare pass!")
    else:
        # assert seqlen == 1
        if is_PerfAI:
            path = "decode_attn"
            generate_run_perfai_script(path)
        if profile_mode != None:
            path = "decode_attn"
            generate_run_profile_script(path)
        net_tpu = SophLlamaAttention(cfg, profile_mode=profile_mode,)
        qkv_tpu = torch.rand(
            (batch, cfg.D * (cfg.HEAD_NUM // cfg.TP + 2 * cfg.KV_HEAD_NUM // cfg.TP)),
            dtype=cfg.DTYPE,
            device=cfg.DEVICE,
        )
        query_tpu, key_tpu, value_tpu = qkv_tpu.split(
            [
                cfg.D * cfg.HEAD_NUM // cfg.TP,
                cfg.D * cfg.KV_HEAD_NUM // cfg.TP,
                cfg.D * cfg.KV_HEAD_NUM // cfg.TP,
            ],
            dim=1,
        )
        query_tpu = query_tpu.view(-1, cfg.HEAD_NUM // cfg.TP, cfg.D)
        kv_view = lambda x: x.view(-1, cfg.KV_HEAD_NUM // cfg.TP, cfg.D)
        key_tpu = kv_view(key_tpu)
        value_tpu = kv_view(value_tpu)
        kv_cache_tpu = (
            torch.rand(
                (cfg.NUM_BLOCKS, cfg.BLOCK_SIZE, cfg.KV_HEAD_NUM // cfg.TP, cfg.D),
                dtype=cfg.DTYPE,
                device=cfg.DEVICE,
            ),
            torch.rand(
                (cfg.NUM_BLOCKS, cfg.BLOCK_SIZE, cfg.KV_HEAD_NUM // cfg.TP, cfg.D),
                dtype=cfg.DTYPE,
                device=cfg.DEVICE,
            ),
        )
        cos_tpu = torch.rand((batch, 1, cfg.D), dtype=cfg.DTYPE, device=cfg.DEVICE)
        sin_tpu = torch.rand((batch, 1, cfg.D), dtype=cfg.DTYPE, device=cfg.DEVICE)
        mask = None
        input_lengths_tpu = torch.tensor(
            [cfg.DECODE_START + seqlen -1] * batch, dtype=torch.int32, device=cfg.DEVICE
        )
        cfg.MAX_BLOCKS = (cfg.DECODE_START + seqlen -1) // cfg.BLOCK_SIZE + 1
        cfg.MAX_SEQLEN= cfg.MAX_BLOCKS * cfg.BLOCK_SIZE
        save_slots_tpu = torch.tensor(
            [[cfg.DECODE_START + seqlen -2 + b * cfg.MAX_SEQLEN] for b in range(batch)],
            dtype=torch.int32,
            device=cfg.DEVICE,
        )
        fetch_slots_tpu = torch.tensor(
            [
                [
                    i
                    for i in range(
                        b * cfg.MAX_SEQLEN,
                        b * cfg.MAX_SEQLEN + cfg.DECODE_START + seqlen -1,
                        cfg.BLOCK_SIZE,
                    )
                ]
                for b in range(batch)
            ],
            dtype=torch.int32,
            device=cfg.DEVICE,
        )
        block_tables_tpu = torch.tensor(
            [
                [
                    i
                    for i in range(
                        b * cfg.MAX_BLOCKS,
                        b * cfg.MAX_BLOCKS + (cfg.DECODE_START + seqlen - 1) // cfg.BLOCK_SIZE + 1,
                        1,
                    )
                ]
                for b in range(batch)
            ],
            dtype=torch.int32,
            device=CFG.DEVICE,
        )
        slot_size_tpu = block_tables_tpu.shape[1]
        output_tpu = torch.empty_like(query_tpu)

        logger.info(
            f"test_parameters:\n{output_tpu.shape}\n{query_tpu.shape=}\n{key_tpu.shape=}\n{value_tpu.shape=}\n\
    {kv_cache_tpu[0].shape=}\n{kv_cache_tpu[1].shape=}\n{cos_tpu.shape=}\n{sin_tpu.shape=}\n{mask=}\
    {input_lengths_tpu.cpu()=}\n{save_slots_tpu.cpu()=}\n{slot_size_tpu=}\n{block_tables_tpu.cpu()=}"
        )
        net_tpu(
            output_tpu,
            query_tpu,
            key_tpu,
            value_tpu,
            kv_cache_tpu,
            cos_tpu,
            sin_tpu,
            input_lengths_tpu.cpu(),
            save_slots_tpu,
            block_tables_tpu,
            mask,
            slot_size_tpu,
            cfg.DECODE_START + seqlen -1,
            cfg.BLOCK_SIZE,
        )


def generate_run_perfai_script(data_source):

    script_content = f"""#!/bin/bash

cd /workspace/PerfAI_Release_20241127

source envsetup.sh

./AutoRunner.sh -c sg2260 -d /workspace/tpu-train/{data_source} -e sg2260

cd /workspace/tpu-train
"""

    with open("/workspace/tpu-train/run_perfai.sh", "w") as script_file:
        script_file.write(script_content)

    os.chmod("/workspace/tpu-train/run_perfai.sh", 0o755)

    script_abs_path = os.path.abspath("/workspace/tpu-train/run_perfai.sh")
    print(
        f"=============== run_perfai.sh Generated and made executable: {script_abs_path} ==============="
    )

def generate_run_profile_script(data_source):

    script_content = f"""#!/bin/bash

mv cdm_profile_data_dev0-0 /workspace/tpu-mlir/cdm_profile_data_dev0-0_{data_source}

cd /workspace/tpu-mlir

source envsetup.sh

tpu_profile.py cdm_profile_data_dev0-0_{data_source} cdm_out_{data_source} --mode time --arch BM1690

cp -rf cdm_out_{data_source} /workspace/tpu-train/

cd /workspace/tpu-train
"""
    with open("/workspace/tpu-train/run_profile.sh", "w") as script_file:
        script_file.write(script_content)

    os.chmod("/workspace/tpu-train/run_profile.sh", 0o755)

    script_abs_path = os.path.abspath("/workspace/tpu-train/run_profile.sh")
    print(
        f"=============== run_profile.sh Generated and made executable: {script_abs_path} ==============="
    )

if __name__ == "__main__":
    # os.environ['FORBID_CMD_EXECUTE']='1'
    # os.environ['DISABLE_CACHE'] = '1'
    is_prefill = args.prefill
    is_w4a16 = args.w4a16
    is_test = args.test if args.test else None
    is_perfAI = args.PerfAI if args.PerfAI else None
    test_case = args.case
    profile_mode = args.profile_mode if args.profile_mode else None
    batch = args.batch
    func_name = (
        f"check_{test_case}"
        + (
            "_w4a16"
            if is_w4a16 and not is_prefill and test_case in ["mlp", "attn_fc", "mmqkv"]
            else ""
        )
        + ("_prefill" if is_prefill and test_case in ["attn"] else "")
    )

    model_name = args.model
    CHAT_WRAPPER_TOKEN_NUM = {"llama2_70b": 27, "llama2_7b":27, "llama3_8b": 27, "llama3_70b":27, "qwen_72b": 19, "qwen_7b":19}
    input_length = 5 + CHAT_WRAPPER_TOKEN_NUM[model_name]+ max(0, args.seq - 5 - CHAT_WRAPPER_TOKEN_NUM[model_name])
    seqlen = input_length if is_prefill or test_case in ["attn"] else 1
    ins_folder = (
        f"prefill_{test_case}"
        if is_prefill
        else f"w4a16_{test_case}" if is_w4a16 else f"decode_{test_case}"
    )
    if is_perfAI:
        if os.path.exists(ins_folder):
            shutil.rmtree(ins_folder)
        os.mkdir(ins_folder)
        os.chdir(ins_folder)
        os.environ["FILE_DUMP_CMD"] = f"{ins_folder}"
    current_module = sys.modules[__name__]
    func = getattr(current_module, func_name)
    cfg = getattr(current_module, f"{model_name}_cfg")
    cfg.TP = args.tp
    if (cfg.INTER_SIZE // cfg.TP) % 256 !=0:
        APPEND_SIZE_PER_TP = 256 - (cfg.INTER_SIZE // cfg.TP) % 256 
        cfg.INTER_SIZE = cfg.INTER_SIZE + APPEND_SIZE_PER_TP * cfg.TP
    cfg.NUM_BLOCKS = max(1024, (math.ceil((args.seq + 128 + CHAT_WRAPPER_TOKEN_NUM[model_name]) / 16) * batch))
    logger.info(
        f"=============== [test_case]: {func_name} =================\n [batch]: {batch}\n [seqlen]: {seqlen}\n CFG:{cfg}\n =============================================="
    )
    with torch.no_grad():
        if is_w4a16 or func_name in ["check_attn", "check_attn_prefill"]:
            func(cfg, batch, seqlen, is_test, is_perfAI, profile_mode)
        else:
            func(cfg, batch, seqlen, is_test, is_perfAI, is_prefill, profile_mode)
        # torch_tpu.tpu.synchronize()
    logger.info(f"=============== finished =================")
