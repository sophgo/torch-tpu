import ctypes as ct
import os
# from tkinter.messagebox import NO
import torch
import torch_tpu
import math
import numpy as np

top=os.path.dirname(__file__)
if  os.environ.get('FORBID_CMD_EXECUTE') is None:
    os.environ['FORBID_CMD_EXECUTE']=str(0)
elif os.environ.get('FORBID_CMD_EXECUTE')=='0':
    del os.environ['FORBID_CMD_EXECUTE']

os.environ['FILE_DUMP_CMD'] ="ins"
os.environ['CMODEL_GLOBAL_MEM_SIZE'] = "34359738368"

class DumpIns:
    def __init__(self, lib_path = os.path.join(top, "../../third_party/firmware/sg2260/libcmodel_firmware.so")) -> None:
        self.libpath = lib_path
        self._lib = ct.cdll.LoadLibrary(lib_path)
    def dump(self, path):
        print("==========" + path + "==========")
        self._lib.set_file_dump_subdir(path.encode())

class ForwardHack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, inp):
        DumpIns().dump(name)
        return  inp

    @staticmethod
    def backward(ctx, arg1):
        return None, arg1

class BackwardHack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, args):
        ctx.name = name
        return args

    @staticmethod
    def backward(ctx, args):
        print("===========================", ctx.name)
        DumpIns().dump(ctx.name)
        return None, args

def Dump_Data(tensor, name, device):
    torch.save(tensor.cpu(), name + "_" + device + '.pt')
    return

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
        if (torch.isinf(ed) or torch.isinf(sr)):
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
        with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by zero
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
                print(f"rel warning at index {idx} exp {exp_tensor[idx]:.20f} got {got_tensor[idx]:.20f}")
            elif abs_warning_mask[idx]:
                print(f"abs warning at index {idx} exp {exp_tensor[idx]:.20f} got {got_tensor[idx]:.20f}")
            total += 1
            if total > max_error_count and not only_warning:
                return -1, total

        return 0, total

    def cmp_result(self, tensor_target, tensor2_result):
            compare_status = self.compare_float(tensor_target.view(-1), tensor2_result.view(-1), False)
            if compare_status == -1:
                print("Error: Too many warnings detected.")
            cos_my = self.cosine_similarity(tensor_target.view(-1), tensor2_result.view(-1))
            euclidean_dist = self.euclidean_similarity(tensor_target.view(-1), tensor2_result.view(-1))
            print("Result : cosine similarity:", cos_my)
            print("Result : euclidean similarity:", euclidean_dist)
            if(cos_my < 0.9 or euclidean_dist < 0.8 or compare_status == -1):
                return False
            return True
            
