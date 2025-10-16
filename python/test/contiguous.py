import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch_tpu
torch.manual_seed(1000)
device = "tpu"

def case1():
    a = torch.range(1, 6).to(torch.float).view(2,3)
    a_tpu = a.to(device)
    import pdb; pdb.set_trace()
    torch.tpu.OpTimer_reset()
    a_tpu = a_tpu.transpose(1,0)
    b_tpu = a_tpu.contiguous()
    torch.tpu.OpTimer_dump()

    import pdb; pdb.set_trace()

def case2():
    a = torch.range(1, 6).to(torch.float).view(2,3)
    a_tpu = a.to(device)
    torch.tpu.OpTimer_reset()
    a_tpu = a_tpu.transpose(1,0)
    b_tpu = a_tpu + 1
    torch.tpu.OpTimer_dump()
    import pdb; pdb.set_trace()

def case3():
    a = torch.empty_strided((2, 3), (1, 3), dtype=torch.float32)
    a.requires_grad = True
    a_c = a.contiguous()
    ones  = torch.range(1, 6).to(torch.float).view(2, 3)
    import pdb; pdb.set_trace()
    a_c.backward(ones)
    import pdb; pdb.set_trace()
    torch.tpu.OpTimer_reset()
    a_tpu = torch.empty_strided((2, 3), (1, 3), dtype=torch.float32, device='tpu')
    a_tpu.requires_grad = True
    a_tpu_c = a_tpu.contiguous()
    ones_tpu  = torch.range(1, 6).to(torch.float).view(2, 3).to('tpu')
    torch.tpu.OpTimer_dump()

    torch.tpu.OpTimer_reset()
    a_tpu_c.backward(ones_tpu)
    torch.tpu.OpTimer_dump()
    import pdb; pdb.set_trace()


def case_mergeSplitpermute_sequence_256(use_fp16 = False):
    device = "tpu"
    batch = 32
    head_num = 12
    hidden_size = 768
    sequence = 256

    inp = torch.randn(batch, sequence, hidden_size * 3)
    inp_tpu = inp.to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    print("=======cpu=====")
    q,k,v = inp.split(hidden_size, -1)
    q1 =q.view(batch, sequence, head_num, hidden_size // head_num)
    q2 = q1.permute(0, 2, 1, 3)
    q3 = q2.transpose(-1,-2)
    q4 = q3.contiguous()
    print(q3.stride())
    print(q4.stride())

    print("=======tpu=====")
    q_tpu,_,__ = inp_tpu.split(hidden_size, -1)
    q1_tpu =q_tpu.view(batch, sequence, head_num, hidden_size // head_num)
    q2_tpu = q1_tpu.permute(0, 2, 1, 3)
    q3_tpu = q2_tpu.transpose(-1,-2)
    q4_tpu = q3_tpu.contiguous()
    print(q3_tpu.stride())
    print(q4_tpu.stride())

    print("=====compare======")
    diff = q4 - q4_tpu.cpu()
    print("diff", torch.max(abs(diff)))

def test_transpose_2d():
    print("\n=== Testing 2D Transpose ===")
    data = torch.arange(0, 8)  # 2x4 tensor

    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int16, torch.int8]:
        print(f"\nTesting dtype: {dtype}")

        a = (data % 128).to(dtype).view(2, 4)

        a_tpu = a.to(device).transpose(0, 1)
        b_tpu = a_tpu.contiguous()

        a_cpu = a.transpose(0, 1).contiguous()

        diff = a_cpu - b_tpu.cpu()
        max_diff = torch.max(abs(diff))
        print(f"Max difference ({dtype}): {max_diff.item()}")

        if dtype.is_floating_point:
            if dtype == torch.float32:
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-6), "float32 test failed"
            elif dtype == torch.float16:
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-3), "float16 test failed"
            else:  # bfloat16
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-2), "bfloat16 test failed"
        else:
            assert torch.equal(a_cpu, b_tpu.cpu()), f"{dtype} test failed"

    print("2D transpose tests passed!")

def test_transpose_3d():
    print("\n=== Testing 3D Transpose ===")
    data = torch.arange(0, 2*4*6)

    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int16, torch.int8]:
        print(f"\nTesting dtype: {dtype}")

        a = (data % 128).to(dtype).view(2, 4, 6)

        a_tpu = a.to(device).transpose(1, 2)
        b_tpu = a_tpu.contiguous()

        a_cpu = a.transpose(1, 2).contiguous()

        diff = a_cpu - b_tpu.cpu()
        max_diff = torch.max(abs(diff))
        print(f"Max difference ({dtype}): {max_diff.item()}")

        if dtype.is_floating_point:
            if dtype == torch.float32:
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-6), "float32 test failed"
            elif dtype == torch.float16:
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-3), "float16 test failed"
            else:  # bfloat16
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-2), "bfloat16 test failed"
        else:
            assert torch.equal(a_cpu, b_tpu.cpu()), f"{dtype} test failed"

    print("3D transpose tests passed!")

def test_transpose_high_dim():
    print("\n=== Testing High-Dimensional Transpose ===")
    data = torch.arange(0, 2*4*1*4*4*8)

    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int16, torch.int8]:
        print(f"\nTesting dtype: {dtype}")

        a = (data % 128).to(dtype).view(2, 4, 1, 4, 4, 8)

        a_tpu = a.to(device).transpose(1, 4)
        b_tpu = a_tpu.contiguous()

        a_cpu = a.transpose(1, 4).contiguous()

        diff = a_cpu - b_tpu.cpu()
        max_diff = torch.max(abs(diff))
        print(f"Max difference ({dtype}): {max_diff.item()}")

        if dtype.is_floating_point:
            if dtype == torch.float32:
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-6), "float32 test failed"
            elif dtype == torch.float16:
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-3), "float16 test failed"
            else:  # bfloat16
                assert torch.allclose(a_cpu, b_tpu.cpu(), atol=1e-2), "bfloat16 test failed"
        else:
            assert torch.equal(a_cpu, b_tpu.cpu()), f"{dtype} test failed"

    print("High-dimensional transpose tests passed!")


if __name__ == "__main__":
    # case1()
    # case2()
    case3()
    # case_mergeSplitpermute_sequence_256(True)
    # test_transpose_2d()
    # test_transpose_3d()
    # test_transpose_high_dim()
