import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")

def case1_split_heads():
    """
    Splits => (result to uncontiguous Tensor) => Permute
    """
    ############## config ###################
    device = "privateuseone"
    batch = 2
    sequence = 8
    hidden_size = 768
    num_heads = 12
    num_of_qkv = 3
    attn_head_size = hidden_size // num_heads
    #########################################
    
    # 1.input
    inp = torch.rand(batch, sequence, hidden_size * num_of_qkv)
    inp_tpu = inp.to(device)

    # 2.split
    q,k,v = inp.split(hidden_size, dim=2)
    q_tpu,k_tpu,v_tpu = inp_tpu.split(hidden_size, dim=2)
    print(q_tpu.is_contiguous())
    print("===2. after split ")
    print("q's ID: ", id(q))
    # print("--cpu")
    # print(q)
    # print("--tpu")
    # print(q_tpu.cpu())
    print(torch.max(abs(q-q_tpu.cpu())))

    # 3.view
    new_shape = q.size()[:-1] + (num_heads, attn_head_size)
    viewed_q = q.view(*new_shape)
    viewed_k = k.view(*new_shape)
    viewed_v = v.view(*new_shape)
    viewed_q_tpu = q_tpu.view(*new_shape)
    viewed_k_tpu = k_tpu.view(*new_shape)
    viewed_v_tpu = v_tpu.view(*new_shape)

    print("===3. after view ")
    print("viewed_q's ID: ", id(viewed_q))
    print("viewd_q.stride = ", viewed_q.stride())
    print("viewd_q_tpu.stride = ", viewed_q_tpu.stride())
    print(viewed_q_tpu.is_contiguous())
    # print("--cpu")
    # print(viewed_q)
    # print("--tpu")
    # print(viewed_q_tpu.cpu())
    print("max diff view_q: ", torch.max(abs(viewed_q-viewed_q_tpu.cpu())))
    print("max diff view_k:", torch.max(abs(viewed_k-viewed_k_tpu.cpu())))
    print("max diff view_v:", torch.max(abs(viewed_v-viewed_v_tpu.cpu())))


    # 4.permute
    permuted_q = viewed_q.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
    permuted_k = viewed_k.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
    permuted_v = viewed_v.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
    permuted_q_tpu = viewed_q_tpu.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
    permuted_k_tpu = viewed_k_tpu.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
    permuted_v_tpu = viewed_v_tpu.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
    print("===4. after permute ")
    print("permuted_q.stride = ", permuted_q.stride())
    print("permuted_q_tpu.stride = ", permuted_q_tpu.stride())
    print("permuted_q's ID: ", id(permuted_q))
    # print("--cpu")
    # print(permuted_q)
    # print("--tpu")
    #print(permuted_q_tpu.cpu())
    print("max diff permuted_q:", torch.max(abs(permuted_q-permuted_q_tpu.contiguous().cpu())))
    print("max diff permuted_k:", torch.max(abs(permuted_k-permuted_k_tpu.contiguous().cpu())))
    print("max diff permuted_v:", torch.max(abs(permuted_v-permuted_v_tpu.contiguous().cpu())))
    print("cpu permuted_q is contiguous:",permuted_q.is_contiguous())
    print("tpu permuted_q is contiguous:",permuted_q_tpu.is_contiguous())



if __name__ == "__main__":
    case1_split_heads()