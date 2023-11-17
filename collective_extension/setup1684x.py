import os
import torch
from setuptools import setup
from torch.utils import cpp_extension
os.path.dirname(os.path.realpath(torch.__file__))
# sources = ["src/ProcessGroupSCCL.cpp", "src/SCCLDeviceFactory.cpp", "src/ProcessGroupSophon.cpp", "src/SophonDeviceFactory.cpp"]
sources = ["src/ProcessGroupSCCL.cpp", "src/SCCLDeviceFactory.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo",
                f"{os.path.dirname(os.path.abspath(__file__))}/../torch_tpu/csrc/core",
                f"{os.path.dirname(os.path.abspath(__file__))}/../torch_tpu/csrc",
                f"{os.path.dirname(os.path.abspath(__file__))}/../../libsophon/bmlib/include",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo_sophon/",
                ]
library_dirs = [f"{os.path.dirname(os.path.realpath(torch.__file__))}/lib",
                f"{os.path.dirname(os.path.abspath(__file__))}/../build/Debug/torch_tpu",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo_sophon/build/sophon",
                ]

libraries = ["torch_tpu"]

module = cpp_extension.CppExtension(
    name="sccl_collectives",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs = library_dirs,
    libraries = libraries,
    extra_compile_args=['-g'],
)

setup(
    name="Sophon-Collectives",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
