import os
import deepspeed
import torch_tpu

torch_tpu_path = os.path.dirname(os.path.abspath(torch_tpu.__file__))
# libsophon_path = os.environ.get('LIBSOPHON_TOP')
tpu_train_path = os.environ.get('TPUTRAIN_TOP')


def sources(self):
    return [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csrc_tpu/adam/cpu_adam.cpp')]


def include_paths(self):
    return [os.path.join(self.torch_tpu_path, 'include/torch_tpu/csrc/core'),
            os.path.join(self.torch_tpu_path, 'include/torch_tpu/csrc'),
            os.path.join(self.torch_tpu_path, 'include'),
            os.path.join(self.torch_tpu_path, '..'),
            os.path.join(self.torch_tpu_path, 'csrc'),
            os.path.join(self.torch_tpu_path, 'csrc/core'),
            os.path.join(self.tpu_train_path, 'sgdnn/include'),
            os.path.join(self.tpu_train_path, 'third_party/bmlib/include')]


def extra_ldflags(self):
    return ['-L' + os.path.join(self.torch_tpu_path, 'lib'), '-ltorch_tpu']


def cxx_args(self):
    args = ['-O0', '-std=c++17', '-g', '-Wl,-z,now',
            '-L' + os.path.join(self.torch_tpu_path, 'lib')]
    return args


def nvcc_args(self):
    return []


deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.torch_tpu_path = torch_tpu_path
# deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.libsophon_path = libsophon_path
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.tpu_train_path = tpu_train_path
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.sources = sources
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.include_paths = include_paths
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.extra_ldflags = extra_ldflags
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.cxx_args = cxx_args
deepspeed.ops.op_builder.cpu_adam.CPUAdamBuilder.nvcc_args = nvcc_args