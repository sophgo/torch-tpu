import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

# sources = ["src/ProcessGroupSCCL.cpp", "src/SCCLDeviceFactory.cpp", "src/ProcessGroupSophon.cpp", "src/SophonDeviceFactory.cpp"]
sources = ["src/ProcessGroupSCCL.cpp", "src/SCCLDeviceFactory.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo",
                f"{os.path.dirname(os.path.abspath(__file__))}/../libtorch_plugin/tpu",
                f"{os.path.dirname(os.path.abspath(__file__))}/../libtorch_plugin",
                f"{os.path.dirname(os.path.abspath(__file__))}/../../libsophon/bmlib/include",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo_sophon/",
                ]
library_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/../libtorch/lib",
                f"{os.path.dirname(os.path.abspath(__file__))}/../libtorch_plugin/build",
                f"{os.path.dirname(os.path.abspath(__file__))}/third_party/gloo_sophon/build/sophon",
                ]

libraries = ["gloo", "libtorch_plugin", "sophon"]

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
