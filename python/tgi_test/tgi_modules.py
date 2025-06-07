import torch
import torch_tpu
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import numpy as np

DEFAULT_DTYPE = torch.float16
DEFAULT_DEVICE = "tpu:0"


@dataclass
class MODEL_CFG:
    HIDDEN_SIZE: int
    INTER_SIZE: int
    HEAD_NUM: int
    KV_HEAD_NUM: int
    D: int = 128
    VOCAB_SIZE: int = 32000
    EPS: float = 1e-6
    TP: int = 1

    DEVICE = DEFAULT_DEVICE
    DTYPE: torch.dtype = DEFAULT_DTYPE

    NUM_BLOCKS: int = 2048
    BLOCK_SIZE: int = 16
    DECODE_START: int = 128
    MAX_SEQLEN: int = 4096

    MMQKV_BIAS: bool = False

def my_batch_expand(A: torch.tensor, batch):
    A_batch = A.shape[0]
    row = A.shape[1]
    col = A.shape[2]
    assert A.dim() == 3
    if A_batch == 1:
        return A.expand(batch, row, col)
    else:
        rep = int(batch / A_batch)
        A = A.reshape(A_batch, 1, row, col)
        A = A.expand(A_batch, rep, row, col)
        return A.reshape(batch, row, col)

def my_matmul(A: torch.tensor, B: torch.tensor):
    print(f"{A.dtype=}, {B.dtype=}")
    print(f"{A.shape=}, {B.shape=}")
    assert (
        (A.dim() == 2 and B.dim() == 2)
        or (A.dim() == 3 and B.dim() == 3)
        or (A.dim() == 3 and B.dim() == 2)
    )

    batch = A.shape[0] if A.dim() == 3 else 1
    l_row = A.shape[1] if A.dim() == 3 else A.shape[0]
    l_col = A.shape[2] if A.dim() == 3 else A.shape[1]
    r_row = B.shape[1] if B.dim() == 3 else B.shape[0]
    r_col = B.shape[2] if B.dim() == 3 else B.shape[1]

    assert l_col == r_row

    result = (
        torch.empty((batch, l_row, r_col), device=A.device, dtype=A.dtype)
        if A.dim() == 3
        else torch.empty((l_row, r_col), device=A.device, dtype=A.dtype)
    )

    if A.dim() == 2 and B.dim() == 2:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = (A[i] * B[:, j]).sum()
    elif A.dim() == 3 and B.dim() == 3:
        # print(f'{A.dtype=}, {B.dtype=}')
        # print(f'{A.shape=}, {B.shape=}')
        if A.shape[0] > B.shape[0]:
            # B = B.expand(batch, r_row, r_col)
            B = my_batch_expand(B, batch)
            print(f"{A.dtype=}, {B.dtype=}")
            print(f"{A.shape=}, {B.shape=}")
        elif A.shape[0] < B.shape[0]:
            A = my_batch_expand(A, B.shape[0])
            # A = A.expand(B.shape[0], l_row, l_col)
        for b in range(result.shape[0]):
            for i in range(result.shape[1]):
                for j in range(result.shape[2]):
                    result[b][i][j] = (A[b][i] * B[b][:, j]).sum()
    else:
        for b in range(result.shape[0]):
            for i in range(result.shape[1]):
                for j in range(result.shape[2]):
                    result[b][i][j] = (A[b][i] * B[:, j]).sum()
    return result


class TensorComparator:
    def __init__(self, delta=1e-1, max_error_count=128):
        self.delta = delta
        self.max_error_count = max_error_count

    @staticmethod
    def cosine_similarity(x, y):
        numerator = torch.sum(x * y)
        sqrt_x = torch.sqrt(torch.sum(torch.pow(x, 2)))
        sqrt_y = torch.sqrt(torch.sum(torch.pow(y, 2)))
        denominator = sqrt_x * sqrt_y
        if denominator.item() == 0.0:
            if sqrt_x.item() == 0.0 and sqrt_y.item() == 0.0:
                return 1.0
            else:
                return 0.0
        return numerator / denominator

    @staticmethod
    def euclidean_similarity(x, y):
        ed = torch.sqrt(torch.sum(torch.pow(x - y, 2)))
        sr = torch.sqrt(torch.sum(torch.pow((x + y) / 2, 2))) + 1e-7
        if torch.isinf(ed) or torch.isinf(sr):
            res = 0.0
        else:
            res = 1 - ed / sr
        return res

    def compare_float(self, exp_tensor, got_tensor, only_warning):

        total = 0
        max_error_count = 128
        delta = 1e-1

        exp_tensor = np.array(exp_tensor)
        got_tensor = np.array(got_tensor)

        # Vectorized computation of absolute differences and relative differences
        abs_diff = np.abs(exp_tensor - got_tensor)
        max_abs_vals = np.maximum(np.abs(exp_tensor), np.abs(got_tensor))
        min_abs_vals = np.minimum(np.abs(exp_tensor), np.abs(got_tensor))
        with np.errstate(divide="ignore", invalid="ignore"):  # Ignore division by zero
            rel_diff = np.where(min_abs_vals < 1e-20, np.inf, abs_diff / min_abs_vals)

        # Mask for values with max absolute value > 1.0
        mask_large_values = max_abs_vals > 1.0
        # Mask for significant relative differences
        mask_rel_diff = rel_diff > delta
        # Mask for significant absolute differences
        mask_abs_diff = abs_diff > delta

        # Combine masks for warnings
        warning_mask = mask_large_values & (mask_rel_diff | (min_abs_vals < 1e-20))
        abs_warning_mask = ~mask_large_values & mask_abs_diff

        # Count warnings and print messages
        for idx in np.where(warning_mask | abs_warning_mask)[0]:
            if warning_mask[idx]:
                print(
                    f"rel warning at index {idx} exp {exp_tensor[idx]:.20f} got {got_tensor[idx]:.20f}"
                )
            elif abs_warning_mask[idx]:
                print(
                    f"abs warning at index {idx} exp {exp_tensor[idx]:.20f} got {got_tensor[idx]:.20f}"
                )
            total += 1
            if total > max_error_count and not only_warning:
                return -1, total

        return 0, total

    def cmp_result(self, tensor_target, tensor2_result):
        # compare_status = self.compare_float(
        #     tensor_target.view(-1), tensor2_result.view(-1), False
        # )
        # if compare_status == -1:
        #     print("Error: Too many warnings detected.")
        cos_my = self.cosine_similarity(tensor_target.view(-1), tensor2_result.view(-1))
        euclidean_dist = self.euclidean_similarity(
            tensor_target.view(-1), tensor2_result.view(-1)
        )
        print("Result : cosine similarity:", cos_my)
        print("Result : euclidean similarity:", euclidean_dist)
        if cos_my < 0.9 or euclidean_dist < 0.8:
            return False
        return True


class SophLlamaAdd(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode ):
        super().__init__()
        self.profile_mode = profile_mode

    def forward(self, hidden_states, residule):
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(4096, self.profile_mode)
        print(f"residule:{residule.cpu()}")
        print(f"hidden_states:{hidden_states.cpu()}")
        print_tensor_info(hidden_states, "hidden_states")
        print_tensor_info(residule, "residule")
        residule += hidden_states
        residule.cpu()
        torch_tpu.tpu.synchronize()
        print(f"residule:{residule.cpu()}")
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return residule


class LlamaAdd(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()

    def forward(self, hidden_states, residule):
        # print_tensor_info(hidden_states, "hidden_states")
        # print_tensor_info(residule, "residule")
        residule += hidden_states
        return residule

def print_tensor_info(tensor, name):
    """Print shape, dtype, and stride of a tensor to a file."""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}:\n")
        print(f"  Shape: {tensor.shape}\n")
        print(f"  Dtype: {tensor.dtype}\n")
        print(f"  Stride: {tensor.stride()}\n")
        # Convert tensor to numpy array for safe printing
        if name in ["save_slots","input_lengths","block_tables"]:
            tensor_np = tensor.cpu()
            print(f"  Values:\n{tensor_np}\n\n")

class SophLlamaRMSNorm(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode):
        super().__init__()
        self.profile_mode = profile_mode
        self.variance_epsilon = CFG.EPS

    def forward(self, hidden_states, weight, output):
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, self.profile_mode)
        print_tensor_info(hidden_states, "hidden_states")
        print_tensor_info(weight, "weight")
        print_tensor_info(output, "output")
        out=torch.ops.my_ops.rmsnorm_forward(
            hidden_states,
            weight,
            None,
            output,
            hidden_states.dim() - 1,
            self.variance_epsilon,
        )
        out.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return out


class LlamaRMSNorm(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()
        self.variance_epsilon = CFG.EPS

    def forward(
        self,
        hidden_states,
        weight,
        output,
    ):
        ms = torch.mean(
            torch.square(hidden_states), dim=hidden_states.dim() - 1, keepdim=True
        )
        rms = torch.sqrt(ms + self.variance_epsilon)
        y = hidden_states / rms
        output = y * weight
        return output


class SophLlamaMMqkv(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode):
        super().__init__()
        self.profile_mode = profile_mode

    def forward(self, hidden_states, weight, bias):
        print(f"{hidden_states.dtype=}, {weight.dtype=}")
        print(f"{hidden_states.shape=}, {weight.shape=}")
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, self.profile_mode)
        output = F.linear(hidden_states, weight, bias)
        output.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return output


class LlamaMMqkv(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()

    def forward(self, hidden_states, weight, bias):
        if bias is not None:
            return my_matmul(hidden_states, weight) + bias
        else:
            return my_matmul(hidden_states, weight)


class SophLlamaMMqkvW4a16(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode):
        super().__init__()
        self.group_size = 128
        self.weight_bits = 4
        self.profile_mode = profile_mode

    def forward(
        self,
        active,
        qweight,
        bias,
        qzeros,
        scales,
    ):
        print(f"{active.dtype=}, {qweight.dtype=}")
        print(f"{active.shape=}, {qweight.shape=}")
        output = active.new_empty((active.shape[0], qweight.shape[0]))
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, self.profile_mode)
        print_tensor_info(active, "active")
        print_tensor_info(qweight, "qweight")
        print_tensor_info(bias, "bias")
        print_tensor_info(scales, "scales")
        print_tensor_info(qzeros, "qzeros")
        print(f"self.group_size: {self.group_size}\n")
        print(f"self.weight_bits: {self.weight_bits}\n")
        print_tensor_info(output, "output")
        out=torch.ops.my_ops.matmul_gptq_forward(
            active,
            qweight,
            bias,
            scales,
            qzeros,
            self.group_size,
            self.weight_bits,
            output,
        )
        out.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return output


class LlamaMMqkvW4a16(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()

    def forward(self, hidden_states, weight, bias):
        if bias is not None:
            return my_matmul(hidden_states, weight) + bias
        else:
            return my_matmul(hidden_states, weight)


class SophLlamaAttentionFC(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode):
        super().__init__()
        self.profile_mode = profile_mode

    def forward(self, hidden_states, weight):
        print(f"{hidden_states.dtype=}, {weight.dtype=}")
        print(f"{hidden_states.shape=}, {weight.shape=}")
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, self.profile_mode)
        output = F.linear(hidden_states, weight, None)
        output.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return output


class LlamaAttentionFC(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()

    def forward(self, hidden_states, weight):
        return my_matmul(hidden_states, weight)


class SophLlamaAttentionFcW4a16(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode):
        super().__init__()
        self.group_size = 128
        self.weight_bits = 4
        self.profile_mode = profile_mode

    def forward(self, active, qweight, qzeros, scales, output):
        print(f"{active.dtype=}, {qweight.dtype=}")
        print(f"{active.shape=}, {qweight.shape=}")
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, self.profile_mode)
        print_tensor_info(active, "active")
        print_tensor_info(qweight, "qweight")
        # print_tensor_info(bias, "bias")
        print_tensor_info(scales, "scales")
        print_tensor_info(qzeros, "qzeros")
        print(f"self.group_size: {self.group_size}\n")
        print(f"self.weight_bits: {self.weight_bits}\n")
        print_tensor_info(output, "output")
        out=torch.ops.my_ops.matmul_gptq_forward(
            active,
            qweight,
            None,
            scales,
            qzeros,
            self.group_size,
            self.weight_bits,
            output,
        )
        out.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return output


class LlamaAttentionFcW4a16(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()

    def forward(self, hidden_states, weight):
        return my_matmul(hidden_states, weight)


class SophLlamaMlp(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode):
        super().__init__()
        self.profile_mode = profile_mode

    def forward(self, hidden_states, w0, w1, w2, output):
        w0, w1 = w0.transpose(-1, -2).contiguous(), w1.transpose(-1, -2).contiguous()
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, False)
        out=torch.ops.my_ops.llama_mlp_forward(hidden_states, w0, w1, w2, None, None, None, None, None, None,output, False)
        out.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return output


class LlamaMlp(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, w0, w1, w2, output):
        r_mm0 = my_matmul(x, w0)
        r_mm1 = my_matmul(x, w1)
        r_mm1 = r_mm1 * self.sigmoid(r_mm1)
        r_tmp = r_mm0 * r_mm1
        output = my_matmul(r_tmp, w2)
        return output


class SophLlamaMlpW4a16(nn.Module):
    def __init__(self, CFG: MODEL_CFG, profile_mode):
        super().__init__()
        self.group_size = 128
        self.bits = 4
        self.profile_mode = profile_mode

    def forward(self, hidden_states, w0, z0, s0, w1, z1, s1, w2, z2, s2, output):
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, False)
        print_tensor_info(w0, "up_qweight")
        print_tensor_info(z0, "up_qzeros")
        print_tensor_info(s0, "up_scales")
        print_tensor_info(w1, "gate_qweight")
        print_tensor_info(z1, "gate_qzeros")
        print_tensor_info(s1, "gate_scales")
        print_tensor_info(w2, "down_qweight")
        print_tensor_info(z2, "down_qzeros")
        print_tensor_info(s2, "down_scales")
        print(f"groupsize: {self.group_size}\n")
        print(f"bits: {self.bits}\n")
        print_tensor_info(output, "output")
        out=torch.ops.my_ops.llama_mlp_gptq_forward(
            hidden_states,
            w0,
            z0,
            s0,
            w1,
            z1,
            s1,
            w2,
            z2,
            s2,
            self.group_size,
            self.bits,
            output,
        )
        out.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()
        return output


class LlamaMlpW4a16(nn.Module):
    def __init__(self, CFG: MODEL_CFG):
        super().__init__()
        self.group_size = 128
        self.bits = 4
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, w0, w1, w2, output):
        r_mm0 = my_matmul(x, w0)
        r_mm1 = my_matmul(x, w1)
        r_mm1 = r_mm1 * self.sigmoid(r_mm1)
        r_tmp = r_mm0 * r_mm1
        output = my_matmul(r_tmp, w2)
        return output


class SophLlamaAttention(nn.Module):
    def __init__(self, CFG: MODEL_CFG, is_prefill=False, profile_mode = 0):
        super().__init__()
        self.softmax_scale = CFG.D**-0.5
        self.is_prefill = is_prefill
        self.profile_mode = profile_mode

    def forward(
        self,
        attn_output,
        query,
        key,
        value,
        kv_cache,
        cos,
        sin,
        input_lengths,
        save_slots,
        block_tables,
        mask,
        slot_size,
        max_s,
        block_size,
    ):
        if self.profile_mode !=None:
            torch.ops.my_ops.enable_profile(40960, self.profile_mode)
        print_tensor_info(attn_output, "attention_output[1]")
        print_tensor_info(query, "query")
        print_tensor_info(key, "key")
        print_tensor_info(value, "value")
        print_tensor_info(kv_cache[0], "kv_cache[0]")
        print_tensor_info(kv_cache[1], "kv_cache[1]")
        print_tensor_info(cos, "cos")
        print_tensor_info(sin, "sin")
        print_tensor_info(input_lengths, "input_lengths")
        print_tensor_info(save_slots, "save_slots")
        print_tensor_info(block_tables, "block_tables")
        print_tensor_info(mask, "mask")
        print(f"block_tables_size_1: {slot_size}\n")
        print(f"max_s: {max_s}\n")
        print(f"block_size: {block_size}\n")
        print(f"softmax_scale: {self.softmax_scale}\n")

        out=torch.ops.my_ops.llama_attention(
            attn_output,
            query,
            key,
            value,
            kv_cache[0],
            kv_cache[1],
            cos,
            sin,
            input_lengths,
            save_slots,
            block_tables,
            mask,
            slot_size,
            max_s,
            block_size,
            self.softmax_scale,
            2 if self.is_prefill else 3,
        )
        out.cpu()
        torch_tpu.tpu.synchronize()
        if self.profile_mode !=None:
            torch.ops.my_ops.disable_profile()


class LlamaAttention(nn.Module):
    def __init__(self, CFG: MODEL_CFG, is_prefill=False):
        super().__init__()
        self.block_size = CFG.BLOCK_SIZE
        self.softmax_scale = CFG.D**-0.5
        self.is_prefill = is_prefill
        self.num_attention_heads = CFG.HEAD_NUM
        self.kv_heads = CFG.KV_HEAD_NUM
        self.d = CFG.D
        self.tp = CFG.TP

    def Rope(self, x, cos, sin, mask_coeff):
        d = x.shape[-1]
        d_half = d // 2
        x_temp = torch.concat((x[..., d_half:], x[..., :d_half]), dim=-1)
        print(
            f"{x.shape=}\n{cos.shape=}\n{x_temp.shape=}\n{mask_coeff.shape=}\n{sin.shape=}"
        )
        x = x * cos + x_temp * mask_coeff * sin
        return x

    def softmax(self, x, dim):
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x_exp = torch.exp(x.float() - x_max)
        x_sum = torch.sum(x_exp.half(), dim=dim, keepdim=True)
        x_softmax = x_exp / x_sum
        print(f"x_softmax{x_softmax}\n")
        return x_softmax

    def forward(
        self,
        Q,
        K,
        V,
        Kcache,
        Vcache,
        cos,
        sin,
        input_length,
        mask,
        save_slots,
        block_tables,
        max_s,
    ):
        Ylist = []

        # Apply RoPE
        seq_len = Q.shape[2]
        mask_coeff = torch.ones(seq_len, dtype=Q.dtype)
        mask_coeff[: seq_len // 2] *= -1.0
        K = self.Rope(K, cos, sin, mask_coeff)
        Q = self.Rope(Q, cos, sin, mask_coeff)
        if not self.is_prefill:
            for batch_id in range(len(input_length)):
                N = input_length[batch_id]
                Kfetch_list = []
                Vfetch_list = []
                cur_slot_num = int(np.ceil(N / self.block_size))

                fetch_tokens = N - 1
                cur_Q = Q[batch_id, :, :].unsqueeze(0)
                cur_K = K[batch_id, :, :].unsqueeze(0)
                cur_V = V[batch_id, :, :].unsqueeze(0)

                for slot_id in range(cur_slot_num):
                    cur_slot = block_tables[batch_id][slot_id] * self.block_size
                    tokens_cur_block = min(fetch_tokens, self.block_size)
                    Kfetch_list.append(
                        Kcache.view(-1, Kcache.shape[-2], Kcache.shape[-1])[
                            cur_slot : cur_slot + tokens_cur_block, :, :
                        ]
                    )
                    Vfetch_list.append(
                        Vcache.view(-1, Vcache.shape[-2], Vcache.shape[-1])[
                            cur_slot : cur_slot + tokens_cur_block, :, :
                        ]
                    )
                    fetch_tokens -= tokens_cur_block

                Kfetch_list.append(cur_K)
                Vfetch_list.append(cur_V)

                Kconcat = torch.concat(Kfetch_list, dim=0)
                Vconcat = torch.concat(Vfetch_list, dim=0)

                # num_attention_heads = Q.shape[1]
                # kv_heads = K.shape[1]
                # d = Q.shape[2]
                Kconcat = Kconcat.repeat(
                    (1, self.num_attention_heads // self.kv_heads, 1)
                )
                Vconcat = Vconcat.repeat(
                    (1, self.num_attention_heads // self.kv_heads, 1)
                )

                res_qk = (
                    my_matmul(
                        cur_Q.view(self.num_attention_heads // self.tp, 1, self.d),
                        Kconcat.permute(1, 2, 0),
                    )
                    * self.softmax_scale
                )
                if mask is not None:
                    res_qk = res_qk * mask
                res_qk = self.softmax(res_qk, dim=2)

                cur_Y = my_matmul(
                    res_qk, Vconcat.permute(1, 0, 2)
                )  # [num_attention_heads, 1, d] = [num_attention_heads, 1, N] @ [num_attention_heads, N, d]
                Ylist.append(cur_Y)

                # cur_save_slot = save_slots[batch_id][0]
                # print(f"{Kcache.shape=}")
                # Kcache.view(-1, Kcache.shape[-2], Kcache.shape[-1])[
                #     cur_save_slot, :, :
                # ] = cur_K
                # Vcache.view(-1, Vcache.shape[-2], Vcache.shape[-1])[
                #     cur_save_slot, :, :
                # ] = cur_V

            Y = torch.concat(Ylist, dim=1).permute(
                1, 0, 2
            )  # [batch, num_attention_heads, d]
        else:
            batch_offset = 0
            # num_attention_heads = Q.shape[1]
            # kv_heads = K.shape[1]
            # d = Q.shape[2]
            for batch_id in range(len(input_length)):
                N = input_length[batch_id]
                Kfetch_list = []
                Vfetch_list = []
                cur_slot_num = (N + self.block_size - 1) // self.block_size

                fetch_tokens = N - 1
                cur_Q = Q[batch_offset : batch_offset + N, :, :].view(
                    N, self.num_attention_heads // self.tp, self.d
                )  # seq_lenxQheadxd
                cur_K = K[batch_offset : batch_offset + N, :, :].view(
                    N, self.kv_heads // self.tp, self.d
                )  # seq_lenxkv_headxd
                cur_V = V[batch_offset : batch_offset + N, :, :].view(
                    N, self.kv_heads // self.tp, self.d
                )  # seq_lenxkv_headxd

                cur_Q = cur_Q.permute(1, 0, 2)  # Qheadxseq_lenxd
                cur_K = cur_K.permute(1, 0, 2)  # kv_headxseq_lenxd
                cur_V = cur_V.permute(1, 0, 2)  # kv_headxseq_lenxd
                print(f"{cur_Q.shape=}")
                print(f"{cur_K.shape=}")
                print(f"{cur_V.shape=}")
                res_qk = my_matmul(cur_Q, cur_K.permute(0, 2, 1)) * self.softmax_scale
                if mask is not None:
                    res_qk = res_qk * mask
                res_qk = self.softmax(res_qk, dim=2)

                cur_Y = my_matmul(res_qk, cur_V)
                Ylist.append(cur_Y)

                if Kcache is not None and Vcache is not None:
                    cur_K = K[batch_offset : batch_offset + N, :, :].view(
                        N, self.kv_heads // self.tp, self.d
                    )  # seq_lenxkv_headxd
                    cur_V = V[batch_offset : batch_offset + N, :, :].view(
                        N, self.kv_heads // self.tp, self.d
                    )  # seq_lenxkv_headxd
                    print(f"{Kcache.shape=}\n")
                    print(f"{Vcache.shape=}\n")
                    Kcache = Kcache.view(-1, Kcache.shape[-2], Kcache.shape[-1])
                    Vcache = Vcache.view(-1, Kcache.shape[-2], Kcache.shape[-1])
                    cur_Kcache = Kcache.view(
                        Kcache.shape[0] // self.block_size,
                        self.block_size,
                        Kcache.shape[1],
                        Kcache.shape[2],
                    )
                    cur_Vcache = Vcache.view(
                        Vcache.shape[0] // self.block_size,
                        self.block_size,
                        Vcache.shape[1],
                        Vcache.shape[2],
                    )
                    seq_len = cur_K.shape[0]
                    num_blocks = (seq_len + self.block_size - 1) // self.block_size
                    for i in range(num_blocks):
                        save_seq_len = min(seq_len, self.block_size)
                        cur_save_slot = save_slots[batch_id][i]
                        cur_Kcache[
                            cur_save_slot // self.block_size, :save_seq_len, :, :
                        ] = cur_K[
                            i * self.block_size : i * self.block_size + save_seq_len,
                            :,
                            :,
                        ]
                        cur_Vcache[
                            cur_save_slot // self.block_size, :save_seq_len, :, :
                        ] = cur_V[
                            i * self.block_size : i * self.block_size + save_seq_len,
                            :,
                            :,
                        ]
                        seq_len -= self.block_size

                    batch_offset += N

            Y = torch.concat(Ylist, dim=1).permute(
                1, 0, 2
            )  # [batch, seq_len, num_attention_heads, d]
        return Y
