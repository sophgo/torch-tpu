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

class LLamaAttention(nn.Module):
    def __init__(self, cos, sin, mask, hidden_size, num_attention_heads, embeddings, softmax_scale):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.num_attention_heads = num_attention_heads
        self.embeddings = embeddings
        self.d = int(hidden_size / num_attention_heads)
        self.kv_heads = int(num_attention_heads / 8)
        self.cos = cos
        self.sin = sin
        self.mask = mask
    
    def Rope(self, x, cos, sin, mask_coeff):
        x_temp = torch.concat((x[..., 64:], x[..., :64]), dim=-1)

        x = x * cos.unsqueeze(1) + x_temp * mask_coeff * sin.unsqueeze(1)
        return x

    def forward(self, Q, K, V, Kcache, Vcache, input_length, save_slots, fetch_slots, attention_mode):
        #Cache : [block_nums*block_size, kv_heads, d]
        #Q : [batch, attention_heads, d]
        #K : [batch, kv_heads, d]
        #V : [batch, kv_heads, d]
        #cos/sin : prefill-[] decode-[batch, d]
        #mask : prefill-[max_s, max_s] decode-None
        #fetch_slots : [batch, slot_size]
        #save_slots : [batch, 1]

        Ylist = []
        block_size = 16
        mask_coeff = torch.ones(128)
        mask_coeff[:64] *= -1.
        K = self.Rope(K, self.cos, self.sin, mask_coeff)
        Q = self.Rope(Q, self.cos, self.sin, mask_coeff)

        if attention_mode == "decode":
            for batch_id in range(len(input_length)):
                N = input_length[batch_id]
                Kfetch_list =[]
                Vfetch_list =[]
                cur_slot_num = (N + block_size - 1 )// block_size

                fetch_tokens = N - 1
                cur_Q = Q[batch_id, :, :] # 1xQheadxd
                cur_K = K[batch_id, :, :].view(1,self.kv_heads, self.d) # 1xkv_headxd
                cur_V = V[batch_id, :, :].view(1,self.kv_heads, self.d) # 1xkv_headxd
                for slot_id in range(cur_slot_num):
                    cur_slot = fetch_slots[batch_id][slot_id]
                    tokens_cur_block = min(fetch_tokens, block_size)
                    Kfetch_list.append(Kcache[cur_slot:cur_slot+tokens_cur_block, :, :])
                    Vfetch_list.append(Vcache[cur_slot:cur_slot+tokens_cur_block, :, :])
                    fetch_tokens -= tokens_cur_block
                Kfetch_list.append(cur_K)
                Vfetch_list.append(cur_V)
                Kconcat = torch.concat(Kfetch_list, dim=0) # Nxkv_headxd
                Vconcat = torch.concat(Vfetch_list, dim=0) # Nxkv_headxd
                
                Kconcat = Kconcat.repeat((1, int(self.num_attention_heads / self.kv_heads), 1)) # Nxnum_attention_headsxd
                Vconcat = Vconcat.repeat((1, int(self.num_attention_heads / self.kv_heads), 1)) # Nxnum_attention_headsxd
                
                res_qk = torch.matmul(cur_Q.view(self.num_attention_heads, 1, self.d), Kconcat.permute(1,2,0)) * self.softmax_scale
                res_qk = F.softmax(res_qk, dim=2)

                cur_Y = torch.matmul(res_qk, Vconcat.permute(1,0, 2))  #[num_attention_heads, 1, d] = [num_attention_heads, 1, N] @ [num_attention_heads, N, d]
                Ylist.append(cur_Y)

                cur_save_slot = save_slots[batch_id][0]
                Kcache[cur_save_slot, :, :] = cur_K
                Vcache[cur_save_slot, :, :] = cur_V

            Y = torch.concat(Ylist, dim=1).permute(1, 0, 2) #[batch, num_attention_heads, d]
        
        elif attention_mode == "prefill":
            batch_offset = 0
            for batch_id in range(len(input_length)):
                N = input_length[batch_id]
                Kfetch_list =[]
                Vfetch_list =[]
                cur_slot_num = (N + block_size - 1 )// block_size

                fetch_tokens = N - 1
                cur_Q = Q[batch_offset:batch_offset+N, :, :].view(N, self.num_attention_heads, self.d) # seq_lenxQheadxd
                cur_K = K[batch_offset:batch_offset+N, :, :].view(N, self.kv_heads, self.d) # seq_lenxkv_headxd
                cur_V = V[batch_offset:batch_offset+N, :, :].view(N, self.kv_heads, self.d) # seq_lenxkv_headxd

                cur_Q = cur_Q.permute(1, 0, 2) # Qheadxseq_lenxd
                cur_K = cur_K.permute(1, 0, 2) # kv_headxseq_lenxd
                cur_V = cur_V.permute(1, 0, 2) # kv_headxseq_lenxd

                res_qk = torch.matmul(cur_Q, cur_K.permute(0, 2, 1)) * self.softmax_scale
                res_qk = F.softmax(res_qk, dim=2)

                cur_Y = torch.matmul(res_qk, cur_V)
                Ylist.append(cur_Y)

                if Kcache is not None and Vcache is not None:
                    cur_K = K[batch_offset:batch_offset+N, :, :].view(N, self.kv_heads, self.d) # seq_lenxkv_headxd
                    cur_V = V[batch_offset:batch_offset+N, :, :].view(N, self.kv_heads, self.d) # seq_lenxkv_headxd
                    cur_Kcache = Kcache.view(Kcache.shape[0]//block_size, block_size, Kcache.shape[1], Kcache.shape[2])
                    cur_Vcache = Vcache.view(Vcache.shape[0]//block_size, block_size, Vcache.shape[1], Vcache.shape[2])
                    seq_len = cur_K.shape[0]
                    num_blocks = (seq_len + block_size - 1) // block_size
                    for i in range(num_blocks):
                        save_seq_len = min(seq_len, block_size)
                        cur_save_slot = save_slots[batch_id][i]
                        cur_Kcache[cur_save_slot//block_size, :save_seq_len, :, :] = cur_K[i*block_size:i*block_size+save_seq_len, :, :]
                        cur_Vcache[cur_save_slot//block_size, :save_seq_len, :, :] = cur_V[i*block_size:i*block_size+save_seq_len, :, :]
                        seq_len -= block_size

                    batch_offset += N

            Y = torch.concat(Ylist, dim=1).permute(1, 0, 2) #[batch, seq_len, num_attention_heads, d]

        return Y


class LLamaAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Kcache, Vcache, cos, sin, mask, input_length, save_slots, fetch_slots, embeddings, attention_mode,
                softmax_scale, max_s, block_size = 16, dropout_rate = 0.0):
        block_size = 16
        if Kcache is not None and Kcache.dim() == 3:
            Kcache = Kcache.view(Kcache.shape[0]//block_size, block_size, Kcache.shape[1], Kcache.shape[2])
        if Vcache is not None and Vcache.dim() == 3:
            Vcache = Vcache.view(Vcache.shape[0]//block_size, block_size, Vcache.shape[1], Vcache.shape[2])
        if cos.dim() == 2:
            cos = cos.view(cos.shape[0], 1, cos.shape[1])
        if sin.dim() == 2:
            sin = sin.view(sin.shape[0], 1, sin.shape[1])

        output = torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)
        if Kcache is not None and Vcache is not None:
            if attention_mode == "decode":
                Qbuffer = None
                Kbuffer  = torch.empty(Kcache.shape, dtype = Kcache.dtype, device = Kcache.device)
                Vbuffer  = torch.empty(Vcache.shape, dtype = Vcache.dtype, device = Vcache.device)
            elif attention_mode == "prefill":
                Qbuffer = torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)
                Kbuffer  = torch.empty(K.shape, dtype = K.dtype, device = K.device)
                Vbuffer  = torch.empty(V.shape, dtype = V.dtype, device = V.device)
            torch.ops.my_ops.llama_attention(output,
                                        Q,
                                        K,
                                        V,
                                        Kcache,
                                        Vcache,
                                        cos,
                                        sin,
                                        input_length,
                                        save_slots,
                                        fetch_slots,
                                        mask,
                                        fetch_slots.size(1) if attention_mode == 'decode' else save_slots.size(1),
                                        max_s,
                                        block_size,
                                        softmax_scale,
                                        3 if attention_mode == 'decode' else 2)

        else:
            torch.ops.my_ops.llama_attention_forward(output,
                                        Q,
                                        K,
                                        V,
                                        cos,
                                        sin,
                                        mask,
                                        max_s,
                                        softmax_scale,
                                        dropout_rate,
                                        len(input_length))
        return output

class LLamaAttentionBlock(nn.Module):
    def __init__(self, w0, w1, w2, softmax_scale, dropout_rate = 0.0):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.softmax_scale = softmax_scale
        self.dropout_rate = dropout_rate

    def forward(self, Q, K, V, Kcache, Vcache, input_length, save_slots, fetch_slots, embeddings, max_s, attention_mode):
        print(f"shape: {Q.shape, K.shape, V.shape}")
        if Kcache is not None and Vcache is not None:
            print(f"{Kcache.shape, Vcache.shape}")
        return LLamaAttentionFunc.apply(Q, K, V, Kcache, Vcache, self.w0, self.w1, self.w2,  input_length, save_slots, fetch_slots, embeddings, attention_mode, self.softmax_scale, max_s)


def check_llama_attention_decode():
    #  TP8: attention_heads 8, hidden_size 1024
    #  TP4: attention_heads 16, hidden_size 2048
    #  TP8: attention_heads 64, hidden_size 8192
    import copy
    attention_mode = "decode"
    batch = 16
    # # TP8
    attention_heads = 8
    k_v_heads = int(attention_heads / 8)
    hidden_size = 1024
    # TP4
    # attention_heads = 16
    # hidden_size = 2048
    # # TP1
    # attention_heads = 64
    # hidden_size = 8192

    d = int(hidden_size/ attention_heads)  # 128
    assert d == 128
    block_size = 16
    max_blocks = 20
    embeddings = 4096 # no use
    softmax_scale = 1. / np.sqrt(128)  # sqrt(128)
    input_length = torch.tensor([10, 1, 28, 15, 14, 13, 12, 11, 25, 3, 5, 7, 2, 4, 6, 8], dtype=torch.int32)
    max_s = input_length.max()
    Ntotal = torch.sum(input_length).item()
    slots_size = (max_s + block_size - 1) // block_size
    save_slots = torch.tensor([[26], [1], [44], [79], [94], [109], [124], [139], 
                               [171], [179], [197], [215], [258], [244], [278], [296]], dtype=torch.int32)
    fetch_slots = torch.tensor([[16, 0],[0, 0],[48, 32],[64, 0], [80, 0], [96, 0], [112, 0], [128, 0],
                                [144, 160], [176, 0], [192, 0], [208, 0], [256, 0], [240, 0], [272, 0], [288, 0]], dtype=torch.int32)

    softmax_scale = 1.0 / 11.313708498


    # Assuming N and param.d are predefined
    print("====param====")
    print(f"batch_size: {batch}")
    print(f"Qheads: {attention_heads}")
    print(f"hidden_size: {hidden_size}")
    print(f"Total Seq_len: {Ntotal}")  # Replace with actual value of N
    print(f"head_dim: {d}")  # Assuming param.d is equivalent to d calculated above
    print("============")



    # init input
    cos = torch.rand((batch, d), requires_grad=False, dtype=torch.float32)
    sin = torch.rand((batch, d), requires_grad=False, dtype=torch.float32)
    Q = torch.rand((batch, attention_heads, d), requires_grad=False, dtype=torch.float32)
    K = torch.rand((batch, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    V = torch.rand((batch, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    Kcache = torch.rand((max_blocks*block_size, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    Vcache = torch.rand((max_blocks*block_size, k_v_heads, d), requires_grad=False, dtype=torch.float32)

    cos_tpu = copy.deepcopy(cos).to(device).half()
    sin_tpu =  copy.deepcopy(sin).to(device).half()
    Q_tpu =  copy.deepcopy(Q).to(device).half()
    K_tpu = copy.deepcopy(K).to(device).half()
    V_tpu = copy.deepcopy(V).to(device).half()
    Kcache_tpu = copy.deepcopy(Kcache).to(device).half()
    Vcache_tpu = copy.deepcopy(Vcache).to(device).half()
    input_length_tpu = copy.deepcopy(input_length).to(device)
    save_slots_tpu = copy.deepcopy(save_slots).to(device)
    fetch_slots_tpu = copy.deepcopy(fetch_slots).to(device)
    # inf_tensor = torch.tensor([-float('inf')], dtype=torch.float32)
    # zero_tensor = torch.tensor([0.0], dtype=torch.float32)
    # blob_mask = torch.empty((max_s, max_s), dtype=torch.half)
    # # Populate the blob_mask tensor
    # for i in range(max_s):
    #     for j in range(max_s):
    #         if j <= i:
    #             blob_mask[i, j] = zero_tensor.to(dtype=torch.half)
    #         else:
    #             blob_mask[i, j] = inf_tensor.to(dtype=torch.half)

    # init model
    net_cpu = LLamaAttention(cos, sin, None, hidden_size, attention_heads, embeddings, softmax_scale)

    net_tpu = LLamaAttentionBlock(cos_tpu, sin_tpu, None, softmax_scale)

    # inference
    print("=====forward======")
    out_cpu = net_cpu(Q, K, V, Kcache, Vcache, input_length, save_slots, fetch_slots, attention_mode)
    if attention_mode == "prefill":
        out_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu,  input_length_tpu, save_slots_tpu, fetch_slots_tpu, embeddings, max_s, attention_mode)
    elif attention_mode == "decode":
        # out_tpu = torch.tensor(1)
        out_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu,  input_length_tpu, save_slots_tpu, fetch_slots_tpu, embeddings, max_s, attention_mode)

    # compare
    comparator = TensorComparator()
    status1 = comparator.cmp_result(out_cpu.detach().contiguous(), out_tpu.cpu().detach().float())
    status2 = comparator.cmp_result(Kcache.detach(), Kcache_tpu.cpu().detach().float())
    status3 = comparator.cmp_result(Vcache.detach(), Vcache_tpu.cpu().detach().float())
    return status1 and status2 and status3

def check_llama_attention_prefill():
    import copy
    attention_mode = "prefill"
    batch = 16
    # # TP8
    attention_heads = 8
    k_v_heads = int(attention_heads / 8)
    hidden_size = 1024

    d = int(hidden_size/ attention_heads)  # 128
    assert d == 128
    block_size = 16
    max_blocks = 20
    embeddings = 4096 # no use
    softmax_scale = 1. / np.sqrt(d)  # sqrt(128)
    input_length = torch.tensor([10, 1, 28, 15, 14, 13, 12, 11, 25, 3, 5, 7, 2, 4, 6, 8], dtype=torch.int32)
    max_s = input_length.max()
    Ntotal = torch.sum(input_length).item()
    slots_size = (max_s + block_size - 1) // block_size
    save_slots = torch.tensor([[16, 0],[0, 0],[48, 32],[64, 0], [80, 0], [96, 0], [112, 0], [128, 0],
                               [144, 160], [176, 0], [192, 0], [208, 0], [256, 0], [240, 0], [272, 0], [288, 0]], dtype=torch.int32)

    # Assuming N and param.d are predefined
    print("====param====")
    print(f"batch_size: {batch}")
    print(f"Qheads: {attention_heads}")
    print(f"hidden_size: {hidden_size}")
    print(f"Total Seq_len: {Ntotal}")  # Replace with actual value of N
    print(f"head_dim: {d}")  # Assuming param.d is equivalent to d calculated above
    print("============")

    # init input
    cos = torch.rand((Ntotal, d), requires_grad=False, dtype=torch.float32)
    sin = torch.rand((Ntotal, d), requires_grad=False, dtype=torch.float32)
    Q = torch.rand((Ntotal, attention_heads, d), requires_grad=False, dtype=torch.float32)
    K = torch.rand((Ntotal, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    V = torch.rand((Ntotal, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    Kcache = torch.rand((max_blocks*block_size, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    Vcache = torch.rand((max_blocks*block_size, k_v_heads, d), requires_grad=False, dtype=torch.float32)

    cos_tpu = copy.deepcopy(cos).to(device).half()
    sin_tpu =  copy.deepcopy(sin).to(device).half()
    Q_tpu =  copy.deepcopy(Q).to(device).half()
    K_tpu = copy.deepcopy(K).to(device).half()
    V_tpu = copy.deepcopy(V).to(device).half()
    Kcache_tpu = copy.deepcopy(Kcache).to(device).half()
    Vcache_tpu = copy.deepcopy(Vcache).to(device).half()
    input_length_tpu = copy.deepcopy(input_length).to(device)
    save_slots_tpu = copy.deepcopy(save_slots).to(device)

    # init model
    net_cpu = LLamaAttention(cos, sin, None, hidden_size, attention_heads, embeddings, softmax_scale)
    net_tpu = LLamaAttentionBlock(cos_tpu, sin_tpu, None, softmax_scale)

    # inference
    print("=====forward======")
    fetch_slots = None
    fetch_slots_tpu = None
    out_cpu = net_cpu(Q, K, V, Kcache, Vcache, input_length, save_slots, fetch_slots, attention_mode)
    if attention_mode == "prefill":
        out_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu,  input_length_tpu, save_slots_tpu, fetch_slots_tpu, embeddings, max_s, attention_mode)
    elif attention_mode == "decode":
        # out_tpu = torch.tensor(1)
        out_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu,  input_length_tpu, save_slots_tpu, fetch_slots_tpu, embeddings, max_s, attention_mode)

    # compare
    comparator = TensorComparator()
    status1 = comparator.cmp_result(out_cpu.detach().contiguous(), out_tpu.cpu().detach().float())
    status2 = comparator.cmp_result(Kcache.detach(), Kcache_tpu.cpu().detach().float())
    status3 = comparator.cmp_result(Vcache.detach(), Vcache_tpu.cpu().detach().float())
    return status1 and status2 and status3

def check_llama_attention_forward():
    import copy
    attention_mode = "prefill"
    batch = 16
    # # TP8
    attention_heads = 8
    k_v_heads = int(attention_heads / 8)
    hidden_size = 1024

    d = int(hidden_size/ attention_heads)  # 128
    assert d == 128
    block_size = 16
    max_blocks = 20
    embeddings = 4096 # no use
    softmax_scale = 1. / np.sqrt(d)  # sqrt(128)
    input_length = torch.tensor([28], dtype=torch.int32)
    max_s = input_length.max()
    Ntotal = torch.sum(input_length).item()
    slots_size = (max_s + block_size - 1) // block_size
    save_slots = None

    # Assuming N and param.d are predefined
    print("====param====")
    print(f"batch_size: {batch}")
    print(f"Qheads: {attention_heads}")
    print(f"hidden_size: {hidden_size}")
    print(f"Total Seq_len: {Ntotal}")  # Replace with actual value of N
    print(f"head_dim: {d}")  # Assuming param.d is equivalent to d calculated above
    print("============")

    # init input
    cos = torch.rand((Ntotal, d), requires_grad=False, dtype=torch.float32)
    sin = torch.rand((Ntotal, d), requires_grad=False, dtype=torch.float32)
    Q = torch.rand((Ntotal, attention_heads, d), requires_grad=False, dtype=torch.float32)
    K = torch.rand((Ntotal, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    V = torch.rand((Ntotal, k_v_heads, d), requires_grad=False, dtype=torch.float32)
    Kcache = None
    Vcache = None

    cos_tpu = copy.deepcopy(cos).to(device).half()
    sin_tpu =  copy.deepcopy(sin).to(device).half()
    Q_tpu =  copy.deepcopy(Q).to(device).half()
    K_tpu = copy.deepcopy(K).to(device).half()
    V_tpu = copy.deepcopy(V).to(device).half()
    Kcache_tpu = None
    Vcache_tpu = None
    input_length_tpu = copy.deepcopy(input_length).to(device)
    save_slots_tpu = None

    # init model
    net_cpu = LLamaAttention(cos, sin, None, hidden_size, attention_heads, embeddings, softmax_scale)
    net_tpu = LLamaAttentionBlock(cos_tpu, sin_tpu, None, softmax_scale)

    # inference
    print("=====forward======")
    fetch_slots = None
    fetch_slots_tpu = None
    out_cpu = net_cpu(Q, K, V, Kcache, Vcache, input_length, save_slots, fetch_slots, attention_mode)
    if attention_mode == "prefill":
        out_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu,  input_length_tpu, save_slots_tpu, fetch_slots_tpu, embeddings, max_s, attention_mode)
    elif attention_mode == "decode":
        # out_tpu = torch.tensor(1)
        out_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu,  input_length_tpu, save_slots_tpu, fetch_slots_tpu, embeddings, max_s, attention_mode)

    # compare
    comparator = TensorComparator()
    status1 = comparator.cmp_result(out_cpu.detach().contiguous(), out_tpu.cpu().detach().float())
    return status1

if __name__ == "__main__":
    status = check_llama_attention_prefill()
    if status == False:
        print(f"[Failed] llama prefill compare failed!")
        sys.exit(255)
    status = check_llama_attention_decode()
    if status == False:
        print(f"[Failed] llama decode compare failed!")
        sys.exit(255)
    status = check_llama_attention_forward()
    if status == False:
        print(f"[Failed] llama forward compare failed!")
        sys.exit(255)