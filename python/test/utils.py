import torch
import torch.nn as nn
import numpy as np
import ctypes as ct

def cos_sim(vector_a, vector_b):
    vector_a = vector_a.reshape(-1)
    vector_b = vector_b.reshape(-1)
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    with np.errstate(invalid='ignore'):
        cos = np.nan_to_num(num / denom)
    sim = 0.5 + 0.5 * cos
    return sim

def get_model_grad(model, save_path="model_grad.npz", save_ = True):
    grad_dict = {}  
    for name, param in model.named_parameters():
        #print(name)
        if isinstance(param.grad, torch.Tensor):
            grad_dict[name] = param.grad.cpu().numpy()
        else:
            print(name, "has no grad")
    if save_:
        np.savez(save_path, grad_dict)
    return grad_dict

def compare_model_grad(model1, model2):
    cpu_grad = get_model_grad(model1, "gpt_cpu.npz", False)
    tpu_grad = get_model_grad(model2, "gpt_tpu.npz", False)
    print(" ======== compare model's parameter grad =======")
    assert(len(cpu_grad.keys()) == len(tpu_grad.keys()))
    for k in cpu_grad.keys():
        c_g = cpu_grad[k]
        t_g = tpu_grad[k]
        diff = c_g - t_g
        print(k, ",max abs diff: ", np.max(abs(diff)), ", max related diff: ")

class Optimer:
    def __init__(self, libpath = "/home/huyu/workspace/tpu-train/libtorch_plugin/build/liblibtorch_plugin.so"):
        self.lib_path = libpath
        self._lib = ct.cdll.LoadLibrary(libpath)

    def reset(self):
        self._lib.tpu_op_timer_reset()

    def dump(self):
        self._lib.tpu_op_timer_dump()

if __name__ == "__main__":
    optimer = Optimer("../../libtorch_plugin/build/liblibtorch_plugin.so")
    optimer.reset()
    optimer.dump()