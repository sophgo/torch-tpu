import torch
import torch.nn as nn
import copy
import math
import numpy as np
import torch_tpu
TPU_CORE_NUM = 1
NPU_NUM = 64

def div_up(x, y):
    return (x + y - 1) // y

def rotate_every_two(x):
    x_swapped = torch.cat((-x[:, :,(int)(x.shape[-1]/2):], x[:, :, :(int)(x.shape[-1]/2)]), dim=-1)
    return x_swapped
def rotate_every_two_inner(x):
    x_swapped = torch.cat((-x[:, (int)(x.shape[-1]/2):], x[:, :(int)(x.shape[-1]/2)]), dim=-1)
    return x_swapped

def flash_attention_backward_single_batch(q, k, v, o, do, l, idx_theta):
    kv_slice_num = div_up(k.shape[0],NPU_NUM)
    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)
    for kv_sec in range(kv_slice_num):
        kv_slice_size = min(k.shape[0] - kv_sec*NPU_NUM, NPU_NUM)
        kv_offset = kv_sec * NPU_NUM
        k_slice = k[kv_offset : kv_offset + kv_slice_size, :]
        v_slice = v[kv_offset : kv_offset + kv_slice_size, :]
        # 1.S=Q*KT  [N,D]@[D,N]->[N,N]
        k_tran = k_slice.transpose(0, 1)
        _S = np.matmul(q, k_tran)
        # 2.S*C [N,N]
        scale = 1.0 / torch.sqrt(torch.tensor(k.shape[1], dtype=torch.float32))
        S = _S  * scale
        # 3.S-L [N, N]
        tmp = S - l
        # 4.exp(S-L) [N, N]
        P = torch.exp(tmp)
        # 5.P_trans = PT
        P_trans = P.transpose(0, 1)
        # 6.dv = P_trans*do [N, D]
        dv = np.matmul(P_trans.detach(), do.detach())
        dV[kv_offset : kv_offset + kv_slice_size, :] = np.matmul(P_trans.detach(), do.detach())
        # 7.dp = do * VT [N, N]@[N,D]
        v_trans = v_slice.transpose(0, 1)
        dP = np.matmul(do, v_trans)
        # 8.dO·O
        _D = torch.mul(do, o)
        # 9.D=rowsum(dO·O)
        D = torch.sum(_D, dim=1).reshape(-1, 1)
        # 10.dp-D [N, N]
        dP_sub_D = torch.sub(dP, D)
        # 11.dS = p*(dp-D) [N, N]
        dS = torch.mul(P, dP_sub_D)
        # 12.dS*scale
        dS = dS*scale
        # 13.dq=dS*K [N, N] @ [N, D] => [N, D]
        dq_ = np.matmul(dS.detach(), k_slice.detach())
        # Rope_bw for dQ
        dQ += dq_
        # 14.dST
        dS_tran = dS.transpose(0, 1)
        # 15.dK = dST*Q [N, D]
        dk_ = np.matmul(dS_tran.detach(), q.detach())
        # Rope_bw for dK
        dk = dk_ * torch.cos(idx_theta[kv_offset : kv_offset + kv_slice_size,:]) - rotate_every_two_inner(dk_) * torch.sin(idx_theta[kv_offset : kv_offset + kv_slice_size,:])
        dK[kv_offset : kv_offset + kv_slice_size, :] = dk
    dQ = dQ * torch.cos(idx_theta) - rotate_every_two_inner(dQ) * torch.sin(idx_theta)
    return dQ, dK, dV

def flash_attention_backward_flatten(q, k, v, o, do, l,  batch, input_lens, num_attention_head, num_kv_head, idx_theta):
    Ntotal = int(torch.sum(input_lens))
    q = q * torch.cos(idx_theta).half() + rotate_every_two(q) * torch.sin(idx_theta).half()
    k = k * torch.cos(idx_theta).half() + rotate_every_two(k) * torch.sin(idx_theta).half()

    # convert [NTotal, num_kv_head, d] to [num_kv_head, NTotal, d]
    q = q.transpose(0, 1)
    o = o.transpose(0, 1)
    do = do.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    # convert [Ntotal, num_attention_head] to [num_attention_head, NTotal]
    l = l.transpose(0, 1)

    idx_theta = idx_theta.transpose(0, 1)
    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)
    qhead_percore = div_up(num_attention_head, TPU_CORE_NUM)
    qhead_secs = div_up(num_attention_head, qhead_percore)
    heads_rep = div_up(num_attention_head, num_kv_head)
    for core_idx in range(qhead_secs):
        if core_idx == qhead_secs - 1:
            qhead_percore = num_attention_head - core_idx * qhead_percore
        for hidx in range(qhead_percore):
            token_offset = 0
            for nidx in range(batch):
                token = input_lens[nidx]
                kv_head_index = core_idx * int(qhead_percore / heads_rep) + hidx // heads_rep
                q_head_index = core_idx * qhead_percore + hidx
                q_per_core = q[q_head_index, token_offset:token_offset + token]
                k_per_core = k[kv_head_index, token_offset:token_offset + token]
                v_per_core = v[kv_head_index, token_offset:token_offset + token]
                o_per_core = o[q_head_index, token_offset:token_offset + token]
                do_per_core = do[q_head_index, token_offset:token_offset + token]
                l_per_core = l[q_head_index, token_offset:token_offset + token].reshape(-1, 1)
                idx_theta_per_core = idx_theta[0, token_offset:token_offset + token]
                _dq, _dk, _dv = flash_attention_backward_single_batch(q_per_core, k_per_core, v_per_core, o_per_core, do_per_core, l_per_core, idx_theta_per_core)
                dQ[q_head_index, token_offset:token_offset + token] = _dq
                dK[kv_head_index, token_offset:token_offset + token] += _dk
                dV[kv_head_index, token_offset:token_offset + token] += _dv
                token_offset += token

    # convert [num_attention_head, NTotal, d] to [NTotal, num_attention_head, d]
    dQ = dQ.transpose(0, 1).reshape(Ntotal, num_q_head, -1)
    # convert [num_kv_head, NTotal, d] to [NTotal, num_kv_head, d]
    dK = dK.transpose(0, 1).reshape(Ntotal, num_kv_head, -1)
    dV = dV.transpose(0, 1).reshape(Ntotal, num_kv_head, -1)
    return dQ, dK, dV

class GQA(torch.nn.Module):
    def __init__(self, num_attention_head, num_kv_head, idx_theta, softmax_scale, input_lengths):
        super(GQA, self).__init__()
        self.num_attention_head = num_attention_head
        self.num_kv_head = num_kv_head
        self.tao = softmax_scale
        self.idx_theta = idx_theta
        self.batch = input_lengths.shape[0]
        self.input_lengths = input_lengths

    def rotate_every_two(self,x):
        x_swapped = torch.cat((-x[:, :,(int)(x.shape[-1]/2):], x[:, :, :(int)(x.shape[-1]/2)]), dim=-1)
        return x_swapped

    def attention(self, q, k, v):
        qk = torch.matmul(q, k.transpose(0, 1))*self.tao
        qk_max = torch.max(qk,dim=1,keepdim=True)[0]
        lse = torch.logsumexp(qk-qk_max, dim=1).reshape(-1, 1)
        l = (lse+qk_max)
        # q @ k^T -> (seq_len, seq_len)
        S = torch.matmul(q, k.transpose(0, 1))
        # P = softmax(S)
        P = torch.softmax(S * self.tao, dim=1)
        # P @ v -> (seq_len, d)
        return torch.matmul(P, v), l

    def forward(self, q, k, v):
        q_rope = q * torch.cos(self.idx_theta) + self.rotate_every_two(q) * torch.sin(self.idx_theta)
        k_rope = k * torch.cos(self.idx_theta) + self.rotate_every_two(k) * torch.sin(self.idx_theta)
        o = torch.zeros_like(q)
        l = torch.zeros((q.shape[0], self.num_attention_head, 1))
        for i in range(self.num_attention_head):
            token_offset = 0
            for nidx in range(self.batch):
                token = self.input_lengths[nidx]
                _q = q_rope[token_offset:token_offset + token, i]
                _k = k_rope[token_offset:token_offset + token, i // (self.num_attention_head // self.num_kv_head)]
                _v = v[token_offset:token_offset + token, i // (self.num_attention_head // self.num_kv_head)]
                _o, _l = self.attention(_q, _k, _v)
                o[token_offset:token_offset + token, i] = _o
                l[token_offset:token_offset + token, i] = _l
                token_offset += token
        return o.reshape(q.shape[0], q.shape[1], -1), l.reshape(q.shape[0], q.shape[1], -1)


def gen_rand_tensor_single(Ntotal, num_q_head, num_kv_head, head_size):
    q = torch.randn(Ntotal, num_q_head, head_size)
    k = torch.randn(Ntotal, num_kv_head, head_size)
    v = torch.randn(Ntotal, num_kv_head, head_size)
    return q, k, v

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

if __name__ == "__main__":

    #for 8core
    num_q_head = 32
    num_kv_head = 32
    #for 1core
    # num_q_head = 4
    # num_kv_head = 4

    device = "tpu:0"
    torch.manual_seed(1000)
    head_dim = 128
    batch = 2
    seq_len = 70
    Ntotal = seq_len*batch
    softmax_scale = head_dim**-0.5
    input_lengths = torch.tensor([seq_len for _ in range(batch)], dtype=torch.int32)

    #1.generate data
    q, k, v = gen_rand_tensor_single(Ntotal, num_q_head, num_kv_head, head_dim)
    q_cpu = q.clone().half()
    k_cpu = k.clone().half()
    v_cpu = v.clone().half()
    q_tpu = q.clone().half().to(device)
    k_tpu = k.clone().half().to(device)
    v_tpu = v.clone().half().to(device)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    idx_theta = gen_rope(Ntotal, head_dim) #[seq_len,1,head_size]
    max_s = input_lengths.max().item()

    #2.cpu forward
    model = GQA(num_q_head, num_kv_head, idx_theta, softmax_scale, input_lengths)
    o, l = model(q, k, v)
    o_cpu = o.clone().half()
    l_cpu = l.clone().half()
    o_tpu = o.clone().half().to(device)
    l_tpu = l.clone().half().to(device)

    #3.torch backward , get loss & grad
    targets = torch.randn(Ntotal, num_q_head, head_dim)*1000
    criterion = nn.MSELoss()
    loss = criterion(o, targets)
    o.retain_grad()
    loss.backward()
    dq_torch = q.grad.clone().half()
    dk_torch = k.grad.clone().half()
    dv_torch = v.grad.clone().half()
    do = o.grad.clone().half()
    do_cpu = do.clone().half()
    do_tpu = do.clone().to(device)

    #4.cpu backward
    dq_cpu, dk_cpu, dv_cpu = flash_attention_backward_flatten(q_cpu, k_cpu, v_cpu, o_cpu, do_cpu, l_cpu, batch, input_lengths, num_q_head, num_kv_head, idx_theta)

    #5.compare cpu_backward with torch_auto_grad
    print(dq_cpu[0,0,:10], dk_cpu[0,0,:10], dv_cpu[0,0,:10])
    print(dq_torch[0,0,:10], dk_torch[0,0,:10], dv_torch[0,0,:10])
    print("cpu vs torch", torch.max(torch.abs(dq_torch - dq_cpu)), torch.max(torch.abs(dk_torch - dk_cpu)), torch.max(torch.abs(dv_torch - dv_cpu)))
    breakpoint()

    #6. tpu backward
    dq_tpu = torch.empty_like(dq_torch).half().to(device)
    dk_tpu = torch.empty_like(dk_torch).half().to(device)
    dv_tpu = torch.empty_like(dv_torch).half().to(device)
    cos_tpu = torch.cos(idx_theta[:,0,:]).view(idx_theta.shape[0],1,-1).contiguous().half().to(device)
    sin_tpu = torch.sin(idx_theta[:,0,:]).view(idx_theta.shape[0],1,-1).contiguous().half().to(device)
    torch.ops.my_ops.llama_attention_backward(q_tpu, k_tpu, v_tpu, o_tpu, do_tpu, l_tpu, dq_tpu, dk_tpu, dv_tpu, cos_tpu, sin_tpu, input_lengths.to(device), softmax_scale)

    #7.compare cpu_backward with torch_auto_grad
    print(dq_tpu.cpu()[0,0,:10], dk_tpu.cpu()[0,0,:10], dv_tpu.cpu()[0,0,:10])
    print(dq_torch[0,0,:10], dk_torch[0,0,:10], dv_torch[0,0,:10])
    print("tpu vs torch",torch.max(torch.abs(dq_tpu.cpu() - dq_torch)), torch.max(torch.abs(dk_tpu.cpu() - dk_torch)), torch.max(torch.abs(dv_tpu.cpu() - dv_torch)))
    breakpoint()
