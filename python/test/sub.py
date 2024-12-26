import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():

    a1 = torch.randint(0, 5, (5, 5), dtype=torch.float32)
    a1_clone = a1.clone()
    a1_tpu = a1.clone().to(device)
    a2 = torch.randint(1 ,5, (5, 5), dtype=torch.float32)
    a2_clone = a2.clone()
    a2_tpu = a2.clone().to(device)


    a1.sub_(a2)
    a1_tpu.sub_(a2_tpu)
    print("***************test_sub_normal begin*****************")
    print("a1: ",a1_clone)
    print("a2 : ", a2_clone)
    print("cpu : ", a1)
    print("tpu : ", a1_tpu.cpu())
    print("***************test_sub_normal end*****************")
import ppl
import ppl.language as pl

@torch_tpu.jit
@ppl.autotiling({'block_w': lambda nargs: nargs['block_w'] // 2 if nargs['block_w'] // 2 > 1 else 1})
@ppl.jit
def sub_kernel(x_ptr,
              y_ptr,
              output_ptr,
              N:pl.constexpr,
              C:pl.constexpr,
              H:pl.constexpr,
              W:pl.constexpr,
              block_w:pl.constexpr):
    core_idx = pl.get_core_index()
    core_num = pl.get_core_num()
    #use pass, can't use return
    if core_idx >= core_num:
      pass

    slice = pl.tiu.cast(W + core_num - 1 // core_num, pl.int32)
    cur_slice = min(slice, (W - slice * core_idx))
    slice_offset = core_idx * slice
    shape = [N, C, H, W]
    x_global = pl.gtensor(shape, pl.GLOBAL, x_ptr)
    y_global = pl.gtensor(shape, pl.GLOBAL, y_ptr)
    o_global = pl.gtensor(shape, pl.GLOBAL, output_ptr)
    mem_shape = [N, C, H, block_w]
    x_tensor = pl.make_tensor(mem_shape, x_ptr.dtype)
    y_tensor = pl.make_tensor(mem_shape, y_ptr.dtype)
    for w_idx in range(0, cur_slice, block_w):
      pl.enable_pipeline()
      tile_w = min(block_w, cur_slice - w_idx)
      x = x_tensor.view([N, C, H, tile_w])
      y = y_tensor.view([N, C, H, tile_w])
      pl.dma.load(x, x_global[:,:,:, slice_offset + w_idx:slice_offset + w_idx + tile_w])
      pl.dma.load(y, y_global[:,:,:, slice_offset + w_idx:slice_offset + w_idx + tile_w])
      out = x - y
      pl.dma.store(o_global[:,:,:, slice_offset + w_idx:slice_offset + w_idx + tile_w], out)

@torch_tpu.jit
@ppl.autotiling({'block_w': lambda nargs: nargs['block_w'] // 2 if nargs['block_w'] // 2 > 1 else 1})
@ppl.jit
def add_kernel(x_ptr,
              y_ptr,
              output_ptr,
              N:pl.constexpr,
              C:pl.constexpr,
              H:pl.constexpr,
              W:pl.constexpr,
              block_w:pl.constexpr):
    core_idx = pl.get_core_index()
    core_num = pl.get_core_num()
    #use pass, can't use return
    if core_idx >= core_num:
      pass

    slice = pl.tiu.cast(W + core_num - 1 // core_num, pl.int32)
    cur_slice = min(slice, (W - slice * core_idx))
    slice_offset = core_idx * slice
    shape = [N, C, H, W]
    x_global = pl.gtensor(shape, pl.GLOBAL, x_ptr)
    y_global = pl.gtensor(shape, pl.GLOBAL, y_ptr)
    o_global = pl.gtensor(shape, pl.GLOBAL, output_ptr)
    mem_shape = [N, C, H, block_w]
    x_tensor = pl.make_tensor(mem_shape, x_ptr.dtype)
    y_tensor = pl.make_tensor(mem_shape, y_ptr.dtype)
    for w_idx in range(0, cur_slice, block_w):
      pl.enable_pipeline()
      tile_w = min(block_w, cur_slice - w_idx)
      x = x_tensor.view([N, C, H, tile_w])
      y = y_tensor.view([N, C, H, tile_w])
      pl.dma.load(x, x_global[:,:,:, slice_offset + w_idx:slice_offset + w_idx + tile_w])
      pl.dma.load(y, y_global[:,:,:, slice_offset + w_idx:slice_offset + w_idx + tile_w])
      out = x + y
      pl.dma.store(o_global[:,:,:, slice_offset + w_idx:slice_offset + w_idx + tile_w], out)

def case2():

    a1 = torch.randint(0, 5, (2, 67, 10, 4800), dtype=torch.int32)
    a1_clone = a1.clone()
    a1_tpu = a1.clone().to(device)
    a2 = torch.randint(1 ,5, (2, 67, 10, 4800), dtype=torch.int32)
    a2_clone = a2.clone()
    a2_tpu = a2.clone().to(device)

    a3 = torch.randint(1 ,5, (2, 67, 10, 4800), dtype=torch.int32)
    a3_clone = a3.clone()
    a3_tpu = a3.clone().to(device)
    a4 = torch.randint(1 ,5, (2, 67, 10, 4800), dtype=torch.int32)
    a4_clone = a4.clone()
    a4_tpu = a4.clone().to(device)

    a1.sub_(a2)
    a1.add_(a3)
    a1.sub_(a4)

    sub_kernel[(1, 8,)](a1_tpu, a2_tpu, a1_tpu, 2, 67, 10, 4800, 4800)
    add_kernel[(1, 8,)](a1_tpu, a3_tpu, a1_tpu, 2, 67, 10, 4800, 4800)
    sub_kernel[(1, 8,)](a1_tpu, a4_tpu, a1_tpu, 2, 67, 10, 4800, 4800)
    '''
    a1_tpu.sub_(a2_tpu)
    a1_tpu.add(a3_tpu)
    a1_tpu.sub(a4_tpu)
    '''
    print("***************test_sub_broadcast begin*****************")
    print("a1: ",a1_clone)
    print("a2 : ", a2_clone)
    print("a3 : ", a3_clone)
    print("a4 : ", a4_clone)
    print("cpu : ", a1)
    '''
    case 1: failed.
    torch_tpu.tpu.synchronize()
    a1_tpu = a1_tpu.cpu()
    '''
    #case2: ok
    a1_tpu = a1_tpu.float().cpu()
    print("tpu : ", a1_tpu)
    print("max value diff : ", torch.max(torch.abs(a1 - a1_tpu)))
    print("***************test_sub_broadcast end*****************")

def case3():

    a1 = torch.randint(0, 5, (5, 5), dtype=torch.int32)
    a1_clone = a1.clone()
    a1_tpu = a1_clone.to(device)
    a2 = torch.tensor(1,dtype=torch.int32)


    a1.sub_(a2)
    a1_tpu.sub_(a2)
    print("***************test_sub_const begin*****************")
    print("a1: ",a1_clone)
    print("a2 : ", a2)
    print("cpu : ", a1)
    print("tpu : ", a1_tpu.cpu())
    print("***************test_sub_const end*****************")

if __name__ == "__main__":
    # case1()
    case2()
    #case3()
