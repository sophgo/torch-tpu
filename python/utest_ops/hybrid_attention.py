import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
from top_utest import TensorComparator
import torch_tpu

torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
tpu_device = "tpu:0"
torch.set_printoptions(profile="full")

import time


class PagedAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Kcache, Vcache, cos, sin, mask, input_length, cache_length, slots, block_tables, mode_tensor,
                softmax_scale, max_s, block_size = 16):
        if cos.dim() == 2:
            cos = cos.view(cos.shape[0], 1, cos.shape[1])
        if sin.dim() == 2:
            sin = sin.view(sin.shape[0], 1, sin.shape[1])
        output = torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)
        # modify in TGI
        if len(mode_tensor) != 0:
            # at leat one batch is prefill
            first_large_index = mode_tensor[0].item()
            # all prefill
            if first_large_index == 0:
                # logger.info(f"all prefill")
                torch.ops.my_ops.paged_attention(output, Q, K, V, Kcache, Vcache,
                                                cos, sin, input_length, cache_length, block_tables, block_tables, mask,
                                                V.size(-1), block_tables.size(1), block_tables.size(1), max_s, block_size, softmax_scale, 2)
            # prefill + decode
            # input_length.shape[0] > 1 and 0 < first_large_index < input_length.shape[0]
            else:
                # logger.info(f"prefill and decode")
                decode_length = torch.sum(input_length[:first_large_index])
                # prefill(save)
                torch.ops.my_ops.paged_attention(output[decode_length:, :],
                                                Q[decode_length:, :], K[decode_length:, :], V[decode_length:, :],
                                                Kcache, Vcache,
                                                cos[decode_length:, :], sin[decode_length:, :],
                                                input_length[first_large_index:], cache_length[first_large_index:],
                                                block_tables[first_large_index:, :], block_tables[first_large_index:, :], mask,V.size(-1),
                                                block_tables.size(1), block_tables.size(1),
                                                max_s, block_size, softmax_scale, 2)
                # decode(save+fetch)
                torch.ops.my_ops.paged_attention(output[:decode_length, :],
                                                Q[:decode_length, :], K[:decode_length, :], V[:decode_length, :],
                                                Kcache, Vcache,
                                                cos[:decode_length, :], sin[:decode_length, :],
                                                input_length[:first_large_index]+cache_length[:first_large_index], None,
                                                slots[:decode_length], block_tables[:first_large_index, :], mask, V.size(-1),
                                                block_tables.size(1), block_tables.size(1),
                                                max_s, block_size, softmax_scale, 3)
        else:
            # all decode
            # all items in input_length are 1
            # logger.info(f"all decode")
            torch.ops.my_ops.paged_attention(output, Q, K, V, Kcache, Vcache,
                                            cos, sin, input_length+cache_length, None, slots, block_tables, mask,
                                            V.size(-1), block_tables.size(1), block_tables.size(1), max_s, block_size, softmax_scale, 3)

        return output

class PagedAttentionBlock(nn.Module):
    def __init__(self, w0, w1, w2, softmax_scale):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.softmax_scale = softmax_scale

    def forward(self, Q, K, V, Kcache, Vcache, input_length, cache_length, slots, block_tables, max_s, mode_tensor):
        print(f"shape: {Q.shape, K.shape, V.shape}")
        if Kcache is not None and Vcache is not None:
            print(f"{Kcache.shape, Vcache.shape}")
        return PagedAttentionFunc.apply(Q, K, V, Kcache, Vcache, self.w0, self.w1, self.w2,  input_length, cache_length, slots, block_tables, mode_tensor, self.softmax_scale, max_s)


class HybridAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Kcache, Vcache, cos, sin, mask, input_length, cache_length, slots, block_tables, mode_tensor,
                softmax_scale, max_s, block_size = 16):
        if cos.dim() == 2:
            cos = cos.view(cos.shape[0], 1, cos.shape[1])
        if sin.dim() == 2:
            sin = sin.view(sin.shape[0], 1, sin.shape[1])

        output = torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)
        # import pdb;pdb.set_trace()
        torch.ops.my_ops.hybrid_attention(output,
                                        mode_tensor,
                                        Q,
                                        K,
                                        V,
                                        Kcache,
                                        Vcache,
                                        cos,
                                        sin,
                                        input_length,
                                        cache_length,
                                        input_length + cache_length,
                                        slots,
                                        block_tables,
                                        mask,
                                        block_tables.size(1),
                                        max_s,
                                        block_size,
                                        softmax_scale,
                                        )
        return output


class HybridAttentionBlock(nn.Module):
    def __init__(self, w0, w1, w2, softmax_scale):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.softmax_scale = softmax_scale

    def forward(self, Q, K, V, Kcache, Vcache, input_length, cache_length, slots, block_tables, max_s, mode_tensor_tpu):
        print(f"shape: {Q.shape, K.shape, V.shape}")
        if Kcache is not None and Vcache is not None:
            print(f"{Kcache.shape, Vcache.shape}")
        return HybridAttentionFunc.apply(Q, K, V, Kcache, Vcache, self.w0, self.w1, self.w2,  input_length, cache_length, slots, block_tables, mode_tensor_tpu, self.softmax_scale, max_s)


def check_prefill_and_decode():
    import copy
    batch = 1
    # # TP8
    attention_heads = 8
    kv_heads = int(attention_heads / 8)
    hidden_size = 1024

    d = int(hidden_size/ attention_heads)  # 128
    assert d == 128
    block_size = 16
    max_blocks = 80
    softmax_scale = 1. / np.sqrt(d)  # sqrt(128)
    # init input lenght and cache length
    cache_length_list = [25, 23, 0]
    input_length_list = [1, 1,  16]

    input_length = torch.tensor(input_length_list, dtype=torch.int32)
    cache_length = torch.tensor(cache_length_list, dtype=torch.int32)
    max_s = (input_length + cache_length).max()
    Ntotal = torch.sum(input_length).item()
    slots_size = (max_s + block_size - 1) // block_size
    block_tables = torch.tensor([[1022, 1023],[1020, 1021],[1018, 1019]], dtype=torch.int32)
    slots = torch.tensor([16377, 16343, 16288, 16289, 16290, 16291, 16292, 16293, 16294, 16295,
                          16296, 16297, 16298, 16299, 16300, 16301, 16302, 16303], dtype=torch.int32)

    # Assuming N and param.d are predefined
    print("====param====")
    print(f"batch_size: {batch}")
    print(f"heads: {attention_heads}")
    print(f"kv_heads: {kv_heads}")
    print(f"hidden_size: {hidden_size}")
    print(f"Total Seq_len: {Ntotal}")  # Replace with actual value of N
    print(f"head_dim: {d}")  # Assuming param.d is equivalent to d calculated above
    print(f"max_s: {max_s}")
    print(f"Ntotal: {Ntotal}")
    print(f"slots_size: {slots_size}")
    print("============")

    # init input
    cos = torch.rand((Ntotal, d), requires_grad=False, dtype=torch.float32)
    sin = torch.rand((Ntotal, d), requires_grad=False, dtype=torch.float32)
    Q = torch.rand((Ntotal, attention_heads, d), requires_grad=False, dtype=torch.float32)
    K = torch.rand((Ntotal, kv_heads, d), requires_grad=False, dtype=torch.float32)
    V = torch.rand((Ntotal, kv_heads, d), requires_grad=False, dtype=torch.float32)
    Kcache = torch.zeros((max_blocks, block_size, kv_heads, d), requires_grad=False, dtype=torch.float32)
    Vcache = torch.zeros((max_blocks, block_size, kv_heads, d), requires_grad=False, dtype=torch.float32)
    attn_mask = torch.triu(torch.full((max_s, max_s), float('-inf'), dtype=torch.float32), diagonal=1)

    # to device--attn_seperate
    cos_sep = copy.deepcopy(cos).to(device).half()
    sin_sep =  copy.deepcopy(sin).to(device).half()
    Q_sep =  copy.deepcopy(Q).to(device).half()
    K_sep = copy.deepcopy(K).to(device).half()
    V_sep = copy.deepcopy(V).to(device).half()
    Kcache_sep = copy.deepcopy(Kcache).to(device).half()
    Vcache_sep = copy.deepcopy(Vcache).to(device).half()
    input_length_sep = copy.deepcopy(input_length)
    cache_length_sep = copy.deepcopy(cache_length)
    block_tables_sep = copy.deepcopy(block_tables).to(device)
    slots_sep = copy.deepcopy(slots).to(device)
    attn_mask_sep = copy.deepcopy(attn_mask).to(device).half()
    # to device--attn_hybrid
    cos_hyb = copy.deepcopy(cos).to(device).half()
    sin_hyb =  copy.deepcopy(sin).to(device).half()
    Q_hyb =  copy.deepcopy(Q).to(device).half()
    K_hyb = copy.deepcopy(K).to(device).half()
    V_hyb = copy.deepcopy(V).to(device).half()
    Kcache_hyb = copy.deepcopy(Kcache).to(device).half()
    Vcache_hyb = copy.deepcopy(Vcache).to(device).half()
    input_length_hyb = copy.deepcopy(input_length)
    cache_length_hyb = copy.deepcopy(cache_length)
    block_tables_hyb = copy.deepcopy(block_tables).to(device)
    slots_hyb = copy.deepcopy(slots).to(device)
    attn_mask_hyb = copy.deepcopy(attn_mask).to(device).half()

    # calculate mode_tensor
    mode_tensor = torch.where(input_length > 1)[0]
    mode_tensor_tpu = copy.deepcopy(mode_tensor).to(device)
    # init model
    attn_seperate = PagedAttentionBlock(cos_sep, sin_sep, attn_mask_sep, softmax_scale)
    attn_hybrid = HybridAttentionBlock(cos_hyb, sin_hyb, attn_mask_hyb, softmax_scale)
    # inference
    print("=====prefill and decode======")
    out_attn_seperate = attn_seperate(Q_sep, K_sep, V_sep, Kcache_sep, Vcache_sep, input_length_sep, cache_length_sep, slots_sep, block_tables_sep, max_s, mode_tensor)
    out_attn_hybrid = attn_hybrid(Q_hyb, K_hyb, V_hyb, Kcache_hyb, Vcache_hyb, input_length_hyb, cache_length_hyb, slots_hyb, block_tables_hyb, max_s, mode_tensor_tpu)
    # compare
    comparator = TensorComparator()
    status = comparator.cmp_result(out_attn_seperate.cpu().detach().float(), out_attn_hybrid.cpu().detach().float(), "output")
    return status

if __name__ == "__main__":
    status = check_prefill_and_decode()
    if status == False:
        print(f"[Failed] hybrid prefill and decode compare failed!\n")
        sys.exit(255)
    else:
        print(f"[Passed] hybrid prefill and decode compare passed!\n")
