import torch
import torch.nn as nn
import copy
import math
import numpy as np
import torch_tpu
import time
import sys

TPU_CORE_NUM = 8
NPU_NUM = 64
device = "tpu:0"
import os
os.environ["CMODEL_FAST_EXEC"] = "1"

# batch=8
# seq_len=512
# q_head = 14
# kv_head = 2
# head_dim = 128


#generate data
def gen_rope(seq_len, head_size):
    theta = torch.arange(0, head_size, 2).float() / head_size
    theta = 1. / 10000 ** theta
    theta = theta.view(1, -1).squeeze()  # (1, d/2)
    seq_idx = torch.arange(seq_len, dtype=theta.dtype)
    seq_idx = seq_idx.view(-1, 1).squeeze()  # (seq_len, 1)
    idx_theta = torch.einsum('i,j->ij', seq_idx, theta)  # (seq_len, d/2)
    idx_theta = idx_theta.view(idx_theta.shape[0],1,-1)
    idx_theta = idx_theta.repeat(1,1,2) # (seq_len, 1, head_size)
    return idx_theta
def gen_rand_tensor_single(Ntotal, num_q_head, num_kv_head, head_size):
    q = torch.randn(Ntotal, num_q_head, head_size)
    k = torch.randn(Ntotal, num_kv_head, head_size)
    v = torch.randn(Ntotal, num_kv_head, head_size)
    return q, k, v

def div_up(x, y):
    return (x + y - 1) // y
def rotate_every_two(x):
    x1, x2 = x.chunk(2, dim=-1)
    x_swapped = torch.cat((-x2, x1), dim=-1)
    return x_swapped
def rotate_every_two_inner(x):
    x_swapped = torch.cat((-x[..., (int)(x.shape[-1]/2):], x[..., :(int)(x.shape[-1]/2)]), dim=-1)
    return x_swapped

def flash_attention_backward_single_batch(q, k, v, o, do, l, mask, cos, sin, softmax_scale, dtype, core_idx, device):
    kv_slice_num = div_up(k.shape[0],NPU_NUM)
    dQ = torch.zeros_like(q).float()
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)
    cos = cos.view(cos.shape[0], -1)
    sin = sin.view(sin.shape[0], -1)
    q = q * cos + rotate_every_two(q) * sin
    for kv_sec in range(kv_slice_num):
        kv_slice_size = min(k.shape[0] - kv_sec*NPU_NUM, NPU_NUM)
        kv_offset = kv_sec * NPU_NUM
        k_slice = k[kv_offset : kv_offset + kv_slice_size, :]
        v_slice = v[kv_offset : kv_offset + kv_slice_size, :]
        mask_slice = mask[:, kv_offset : kv_offset + kv_slice_size]
        cos_slice = cos[kv_offset : kv_offset + kv_slice_size, :]
        sin_slice = sin[kv_offset : kv_offset + kv_slice_size, :]
        k_slice = k_slice * cos_slice + rotate_every_two(k_slice) * sin_slice
        # 1.S=Q*KT  [N,D]@[D,N]->[N,N]
        k_tran = k_slice.transpose(0, 1).contiguous()
        _S = torch.matmul(q.float(), k_tran.float()).float()

        # 2.S*C [N,N]
        scale = softmax_scale
        # print("scale", scale)
        S = _S  * scale + mask_slice.float()
        # 3.S-L [N, N]
        tmp = S - l.float()
        # 4.exp(S-L) [N, N]
        P = torch.exp(tmp)
        # 5.P_trans = PT
        P_trans = P.transpose(0, 1).contiguous().to(dtype)
        do = do.detach()
        # 6.dv = P_trans*do [N, D]
        dv = torch.matmul(P_trans.detach(), do)
        # 7.dp = do * VT [N, N]@[N,D]
        v_trans = v_slice.transpose(0, 1).contiguous()
        dP = torch.matmul(do.float(), v_trans.float())#[N, slice]  #>> 对数值影响很大，使用fp32 mm, 0.925->0.90
        # 8.dO·O
        _D = torch.mul(do.float(), o.float())
        # 9.D=rowsum(dO·O)
        D = torch.sum(_D, dim=-1).reshape(-1, 1) #[N,1]
        # 10.dp-D [N, N]
        dP_sub_D = torch.sub(dP, D)
        # 11.dS = p*(dp-D) [N, N]
        dS = torch.mul(P, dP_sub_D)#[N, slice]
        # 12.dS*scale
        dS = dS * scale
        # 13.dq=dS*K [N, N] @ [N, D] => [N, D]
        dq_ = torch.matmul(dS.detach(), k_slice.detach().float())
        # update dQ
        dQ += dq_
        # 14.dST
        dS_tran = dS.transpose(0, 1)
        # 15.dK = dST*Q [N, D]
        dk_ = torch.matmul(dS_tran.contiguous(), q.float()).to(dtype) ##device中会出现nan
        # Rope_bw for dK
        dk = dk_ * cos_slice - rotate_every_two_inner(dk_) * sin_slice
        dk = dk

        dK[kv_offset : kv_offset + kv_slice_size, :] = dk
        dV[kv_offset : kv_offset + kv_slice_size, :] = dv
    dQ = dQ.to(dtype)
    dQ = dQ * cos - rotate_every_two_inner(dQ) * sin
    return dQ, dK, dV

def flash_attention_backward_flatten(q, k, v, o, do, l,  batch, input_lens, num_attention_head, num_kv_head, mask, cos, sin, softmax_scale, dtype, device):

    Ntotal = int(torch.sum(input_lens.cpu()))
    
    # dtype = torch.float32
    # device = torch.device("cpu")
    
    # convert [NTotal, num_kv_head, d] to [num_kv_head, NTotal, d]
    q = q.transpose(0, 1).contiguous().to(device).to(dtype)
    k = k.transpose(0, 1).contiguous().to(device).to(dtype)
    v = v.transpose(0, 1).contiguous().to(device).to(dtype)
    o = o.transpose(0, 1).contiguous().to(device).to(dtype)
    do = do.transpose(0, 1).contiguous().to(device).to(dtype)
    cos = cos.to(device).to(dtype)
    sin = sin.to(device).to(dtype)
    mask = mask.to(device).to(dtype)
    # convert [Ntotal, num_attention_head] to [num_attention_head, NTotal]
    l = l.transpose(0, 1).contiguous().to(device)

    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)
    qhead_percore = div_up(num_attention_head, TPU_CORE_NUM)
    qhead_secs = div_up(num_attention_head, qhead_percore)
    heads_rep = div_up(num_attention_head, num_kv_head)
    for core_idx in range(qhead_secs):
        if core_idx == qhead_secs - 1:
            qhead_percore_cur = num_attention_head - core_idx * qhead_percore
        else:
            qhead_percore_cur = qhead_percore
        for hidx in range(qhead_percore_cur):
            kv_head_index = (core_idx * qhead_percore + hidx) // heads_rep
            token_offset = 0
            for nidx in range(batch):
                token = input_lens[nidx]
                #kv_head_index = core_idx * int(qhead_percore / heads_rep) + hidx // heads_rep
                q_head_index = core_idx * qhead_percore + hidx
                q_per_core = q[q_head_index, token_offset:token_offset + token]
                k_per_core = k[kv_head_index, token_offset:token_offset + token]
                v_per_core = v[kv_head_index, token_offset:token_offset + token]
                o_per_core = o[q_head_index, token_offset:token_offset + token]
                do_per_core = do[q_head_index, token_offset:token_offset + token]
                l_per_core = l[q_head_index, token_offset:token_offset + token].reshape(-1, 1)
                mask_per_batch = mask[token_offset:token_offset + token]
                cos_per_core = cos
                sin_per_core = sin
                _dq, _dk, _dv = flash_attention_backward_single_batch(q_per_core, k_per_core, v_per_core, o_per_core, do_per_core, l_per_core, mask_per_batch, cos_per_core, sin_per_core, softmax_scale, dtype, core_idx, device)
                dQ[q_head_index, token_offset:token_offset + token] = _dq
                dK[kv_head_index, token_offset:token_offset + token] += _dk
                dV[kv_head_index, token_offset:token_offset + token] += _dv
                token_offset += token

    # convert [num_attention_head, NTotal, d] to [NTotal, num_attention_head, d]
    dQ = dQ.transpose(0, 1).reshape(Ntotal, num_attention_head, -1)
    # convert [num_kv_head, NTotal, d] to [NTotal, num_kv_head, d]
    dK = dK.transpose(0, 1).reshape(Ntotal, num_kv_head, -1)
    dV = dV.transpose(0, 1).reshape(Ntotal, num_kv_head, -1)
    return dQ, dK, dV

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    Ntotal, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :].expand(Ntotal, num_key_value_heads, n_rep,  head_dim)
    return hidden_states.reshape(Ntotal, num_key_value_heads * n_rep, head_dim)
def rotate_every_two(x):
    x_swapped = torch.cat((-x[..., (int)(x.shape[-1]/2):], x[..., :(int)(x.shape[-1]/2)]), dim=-1)
    return x_swapped.contiguous()

def attention(batch, q_device, k_device, v_device, cos, sin, mask_tpu, softmax_scale):
    Ntotal, q_head, head_dim = q_device.shape
    _, kv_head, _ = k_device.shape
    seq_len = Ntotal // batch

    scale = softmax_scale
    k_device = repeat_kv(k_device, q_head//kv_head)
    v_device = repeat_kv(v_device, q_head//kv_head)
    mask_tpu = mask_tpu.reshape(batch, seq_len, seq_len).unsqueeze(1).float().cpu()
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q_device2 = q_device.reshape(batch, seq_len, q_head, head_dim)
    k_device2 = k_device.reshape(batch, seq_len, q_head, head_dim)
    v_device2 = v_device.reshape(batch, seq_len, q_head, head_dim)
    q_device2 = q_device2 * cos + rotate_every_two(q_device2) * sin
    k_device2 = k_device2 * cos + rotate_every_two(k_device2) * sin

    q_device2 = q_device2.transpose(1, 2).float().cpu()
    k_device2 = k_device2.transpose(1, 2).float().cpu()
    v_device2 = v_device2.transpose(1, 2).float().cpu()

    qkt = torch.matmul(q_device2, k_device2.transpose(-1, -2)) * scale + mask_tpu

    max_qkt = torch.max(qkt, dim=-1, keepdim=True)[0]
    lse = torch.log(torch.sum(torch.exp(qkt - max_qkt), dim=-1, keepdim=True)) + max_qkt
    lse = lse.transpose(1, 2).reshape(batch*seq_len, q_head).unsqueeze(-1)

    prob = torch.softmax(qkt, dim=-1)
    out = torch.matmul(prob, v_device2).transpose(1, 2)
    out = out.reshape(batch*seq_len, q_head, head_dim)
    return out, lse


def attn_test(dtype):
    assert dtype in [torch.float16, torch.bfloat16]
    num_q_head = 14
    num_kv_head = 2

    head_dim = 128
    softmax_scale =  head_dim**-0.5  #head_dim**-0.5
    batch = 8
    seq_len = 512
    Ntotal = seq_len*batch
    input_lengths = torch.tensor([seq_len]*batch, dtype=torch.int32)
    mask_max = input_lengths.max().item()

    q, k, v = gen_rand_tensor_single(Ntotal, num_q_head, num_kv_head, head_dim)
    q_device = copy.deepcopy(q.detach()).to(device).to(dtype).requires_grad_(True)
    k_device = copy.deepcopy(k.detach()).to(device).to(dtype).requires_grad_(True)
    v_device = copy.deepcopy(v.detach()).to(device).to(dtype).requires_grad_(True)

    mask = torch.triu(torch.full((seq_len, seq_len), float('-10000.0'), dtype=dtype), diagonal=1).to(dtype)
    mask_cpu = mask.repeat(batch, 1, 1).view(batch, seq_len, seq_len)
    mask_tpu = mask_cpu.clone().view(batch * seq_len, seq_len).to(device)# attn fwd/bwd fuse op need 2d tensor [batch * seq_len, seq_len]

    idx_theta = gen_rope(mask_max, head_dim)
    cos_cpu = torch.cos(idx_theta).view(seq_len, 1, head_dim).contiguous()
    sin_cpu = torch.sin(idx_theta).view(seq_len, 1, head_dim).contiguous()


    # cpu attention fwd
    q_cpu = q.requires_grad_(True)
    k_cpu = k.requires_grad_(True)
    v_cpu = v.requires_grad_(True)
    out_cpu, lse_cpu = attention(batch, q_cpu, k_cpu, v_cpu, cos_cpu, sin_cpu, mask_cpu, softmax_scale)

    # tpu attention fwd
    cos_tpu = cos_cpu.to(dtype).to(device) # attn fwd/bwd fuse op need 3d tensor [seq_len, 1, head_dim]
    sin_tpu = sin_cpu.to(dtype).to(device)
    out_fused = torch.zeros_like(out_cpu.to(dtype)).to(device)
    lse_fused = torch.zeros_like(lse_cpu).to(device)
    torch.ops.my_ops.llama_attention_forward(out_fused, q_device, k_device, v_device, cos_tpu, sin_tpu, mask_tpu, lse_fused, input_lengths.cpu(), seq_len, softmax_scale, 0.0, batch)
    torch.tpu.synchronize()
    print(f" attention forward done")

    # cpu backward
    grad_out_cpu = torch.randn_like(out_cpu).requires_grad_(True)
    out_cpu.requires_grad_(True)
    out_cpu.backward(grad_out_cpu)
    grad_q_cpu = q_cpu.grad.clone().cpu().float()
    grad_k_cpu = k_cpu.grad.clone().cpu().float()
    grad_v_cpu = v_cpu.grad.clone().cpu().float()

    # tpu backward
    grad_out_tpu = grad_out_cpu.detach().clone().to(dtype).to(device)
    dq_tpu = torch.zeros_like(q_device)
    dk_tpu = torch.zeros_like(k_device)
    dv_tpu = torch.zeros_like(v_device)
    torch.ops.my_ops.llama_attention_backward(q_device, k_device, v_device, out_fused, grad_out_tpu, lse_fused, dq_tpu, dk_tpu, dv_tpu, cos_tpu, sin_tpu, mask_tpu, input_lengths.to(device), seq_len, softmax_scale)
    torch.tpu.synchronize()
    print(f" attention backward done")

    # tpu backward (pytroch implement for debug)
    # dq_tpu2, dk_tpu2, dv_tpu2 = flash_attention_backward_flatten(q_device, k_device, v_device, out_fused, grad_out_tpu, lse_fused, batch, input_lengths, num_q_head, num_kv_head, mask_tpu, cos_tpu, sin_tpu, softmax_scale, dtype, device)

    # compare results
    # compare fwd: out & lse
    from top_utest import TensorComparator
    comparator = TensorComparator()
    status_lse = comparator.cmp_result(lse_cpu.cpu().float().detach().view(-1), lse_fused.cpu().float().detach().view(-1), "lse_ops")
    status_out = comparator.cmp_result(out_cpu.cpu().float().detach().view(-1), out_fused.cpu().float().detach().view(-1), "out_ops")
    print(f"[fwd] status_lse {status_lse}, status_out {status_out}")

    # compare bwd: grad_q, grad_k, grad_v
    status_dq = comparator.cmp_result(grad_q_cpu.detach().view(-1), dq_tpu.cpu().float().view(-1), "q_cpu")
    status_dk = comparator.cmp_result(grad_k_cpu.detach().view(-1), dk_tpu.cpu().float().view(-1), "k_cpu")
    status_dv = comparator.cmp_result(grad_v_cpu.detach().view(-1), dv_tpu.cpu().float().view(-1), "v_cpu")
    print(f"[bwd tpu_kernel] status_dq {status_dq}, status_dk {status_dk}, status_dv {status_dv}")
    
    # status_q2 = comparator.cmp_result(grad_q_cpu.detach().view(-1), dq_tpu2.cpu().detach().float().view(-1), "q_cpu")
    # status_k2 = comparator.cmp_result(grad_k_cpu.detach().view(-1), dk_tpu2.cpu().detach().float().view(-1), "k_cpu")
    # status_v2 = comparator.cmp_result(grad_v_cpu.detach().view(-1), dv_tpu2.cpu().detach().float().view(-1), "v_cpu")
    # print(f"[bwd pytorch] status_q {status_q2}, status_k {status_k2}, status_v {status_v2}")

    status_fwd = status_lse and status_out
    status_bwd = status_dq and status_dk and status_dv
    return status_fwd, status_bwd

if __name__ == "__main__":
    if os.environ['CHIP_ARCH'] in ['bm1684x']:
        print(f'Skip test for this arch')
        sys.exit(0)

    seed = time.time()
    torch.manual_seed(seed)
    status_fwd, status_bwd = attn_test(torch.float16)
    if not status_fwd or not status_bwd:
        print(f"attention forward/backward test failed, please check the fwd/bwd result (seed {seed})")
        sys.exit(255)
    print("attention forward/backward test passed")

