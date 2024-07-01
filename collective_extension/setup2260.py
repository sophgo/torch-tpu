import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

build_type = "Release"
if "TPUTRAIN_DEBUG" in os.environ and os.environ["TPUTRAIN_DEBUG"] == "ON":
    build_type = "Debug"

sources = ["src/ProcessGroupSophon.cpp", "src/SophonDeviceFactory.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/",
                f"{os.path.dirname(os.path.abspath(__file__))}/../torch_tpu/csrc/core",
                f"{os.path.dirname(os.path.abspath(__file__))}/../",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo_sophon/",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo_sophon/sophon",
                f"{os.path.dirname(os.path.abspath(__file__))}/../third_party/tpuDNN/include",
                f"{os.path.dirname(os.path.abspath(__file__))}/../third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/include/"
                ]
library_dirs = [f"{os.path.dirname(os.path.realpath(torch.__file__))}/lib",
                f"{os.path.dirname(os.path.realpath(__file__))}/lib",
                f"{os.path.dirname(os.path.abspath(__file__))}/../build/torch-tpu/torch_tpu/",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo_sophon/build/sophon",
                f"{os.path.dirname(os.path.abspath(__file__))}/../third_party/tpuDNN/{os.environ.get('CHIP_ARCH', None)}_lib",
                ]
libraries = ["torch_tpu.sg2260", "sophon", "tpudnn"]

module = cpp_extension.CppExtension(
    name="sccl",
    sources=sources,
    include_dirs=include_dirs,
    library_dirs = library_dirs,
    libraries = libraries,
    extra_compile_args=['-g','-fPIC'],
    extra_ldflags=['-Wl,--whole-archive']
)

setup(
    name="sccl",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
