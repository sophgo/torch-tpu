import os
import sys
import subprocess
import platform
import traceback
import multiprocessing
import glob
import shutil
import re
from pathlib import Path

from sysconfig import get_paths

import distutils.command.clean
from distutils.version import LooseVersion
from distutils.command.build_py import build_py

from setuptools import setup, Extension, distutils
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SGDNN_PATH = os.path.join(BASE_DIR, "sgdnn")
SGAPI_STRUCT_PATH = os.path.join(BASE_DIR, "common")
THIRD_PARTY_PATH = os.path.join(BASE_DIR, "third_party")
TPUV7_RUNTIME_PATH = os.path.join(THIRD_PARTY_PATH, "tpuv7_runtime")

VERSION = '2.1.0.post1'
SOC_CROSS = os.environ.get("SOC_CROSS_MODE", None)
SOC_CROSS = True if SOC_CROSS == "ON" else False
CROSS_TOOLCHAINS= os.environ.get("CROSS_TOOLCHAINS", None)
if SOC_CROSS:
    os.environ["CC"] = f"{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc"
    os.environ["CXX"] = f"{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"

def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None

def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    "Returns cmake command."
    cmake_command = 'cmake'
    if platform.system() == 'Windows':
        return cmake_command
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.12.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.12.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.12.0 found')

def get_build_type():
    build_type = 'Release'
    if os.getenv('TPUTRAIN_DEBUG', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
        build_type = 'Debug'

    return build_type

def python_path_dir():
    if SOC_CROSS:
        return os.path.join(CROSS_TOOLCHAINS, "Python-3.8.2/python_3.8.2/include/python3.8")
    return get_paths()['include']

def get_pytorch_dir():
    if SOC_CROSS:
        print(os.path.join(CROSS_TOOLCHAINS, "torchwhl/torch"))
        return os.path.join(CROSS_TOOLCHAINS, "torchwhl/torch")
    try:
        import torch
        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)

def CppExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.
    '''
    pytorch_dir = get_pytorch_dir()
    temp_include_dirs = kwargs.get('include_dirs', [])
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include'))
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include/torch/csrc/api/include'))
    if SOC_CROSS:
        temp_include_dirs.append(os.path.join(CROSS_TOOLCHAINS, "Python-3.8.2/python_3.8.2/include/python3.8"))
    kwargs['include_dirs'] = temp_include_dirs

    temp_library_dirs = kwargs.get('library_dirs', [])
    temp_library_dirs.append(os.path.join(pytorch_dir, 'lib'))
    if SOC_CROSS:
        temp_library_dirs.append(os.path.join(CROSS_TOOLCHAINS, "Python-3.8.2/python_3.8.2/lib/python3.8"))
    kwargs['library_dirs'] = temp_library_dirs

    temp_runtime_library_dirs = kwargs.get('runtime_library_dir', [])
    temp_runtime_library_dirs.append(os.path.join(pytorch_dir, 'lib'))
    kwargs['runtime_library_dirs'] = temp_runtime_library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return Extension(name, sources, *args, **kwargs)

class ExtBase:
    def get_package_dir(self):
        package_dir = self.distribution.package_dir
        package_dir = os.path.join(package_dir.get(''), 'torch_tpu') if package_dir \
            else os.path.join(BASE_DIR, 'torch_tpu')
        package_dir = os.path.abspath(package_dir)
        return package_dir

class CPPLibBuild(build_clib, ExtBase, object):
    def run(self):
        release_mode = os.getenv('RELEASE_MODE') == 'ON'
        if not release_mode:
            self.build()
        else:
            # build bmlib
            os.environ['CHIP_ARCH'] = 'bm1684x'
            self.build()
            # build tpuv7_runtime
            os.environ['CHIP_ARCH'] = 'sg2260'
            self.build()

    def build(self):
        cmake = get_cmake_command()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake

        arch = os.environ.get('CHIP_ARCH')
        build_dir = os.path.join(BASE_DIR, 'build', f'firmware_{arch}')
        src_dir = os.path.join(BASE_DIR, 'firmware_core')
        args = [self.cmake, src_dir]
        def build_firmware():
            os.makedirs(build_dir, exist_ok=True)
            subprocess.check_call(args, cwd=build_dir, env=os.environ)
            subprocess.check_call(['make', '-j'], cwd=build_dir, env=os.environ)

        build_fw = True
        if arch == 'sg2260':
            tc_path = os.environ.get('RISCV_TOOLCHAIN')
        elif arch == 'bm1684x':
            tc_path = os.environ.get('ARM_TOOLCHAIN')
        if not tc_path or not os.path.exists(tc_path):
            build_fw = False

        if build_fw:
            build_firmware()
            fw_path = os.path.join(build_dir, 'libfirmware.so')

        build_dir += '_cmodel'
        if not build_fw:
            fw_path = os.path.join(build_dir, 'libfirmware.so')
        args.append('-DUSING_CMODEL=ON')
        build_firmware()

        package_dir = self.get_package_dir()
        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DPYTHON_INCLUDE_DIR=' + python_path_dir(),
            '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir(),
            '-DBUILD_LIBTORCH=0',
            f'-DKERNEL_MODULE_PATH={fw_path}',
            f'-DCMAKE_INSTALL_PREFIX={package_dir}'
            ]
        build_dir = os.path.join(BASE_DIR, 'build/torch-tpu')
        os.makedirs(build_dir, exist_ok=True)
        build_args = ['-j', str(8)]
        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ)
        subprocess.check_call(['make'] + build_args, cwd=build_dir, env=os.environ)
        subprocess.check_call(['make', 'install/strip'], cwd=build_dir, env=os.environ)

class Build(build_ext, ExtBase, object):

    def run(self):
        self.run_command('build_clib')
        package_dir = self.get_package_dir()
        sym_fn = os.path.join(package_dir, 'lib/libtorch_tpu.so')
        if not os.path.exists(sym_fn):
            lib_fn = glob.glob(os.path.join(package_dir, 'lib/*libtorch_tpu*.so'))[0]
            os.symlink(lib_fn, sym_fn)

        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, f"build/torch-plugin"))
        self.library_dirs.append(
            os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/lib")))
        super(Build, self).run()

        os.unlink(sym_fn)

    def finalize_options(self):
        build_ext.finalize_options(self)
        if SOC_CROSS:
            self.plat_name = "aarch64"  # Example for ARM64

    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        if 'x86_64-linux-gnu' in filename and SOC_CROSS:
            return filename.replace('x86_64-linux-gnu', 'aarch64-linux-gnu')
        return filename

class InstallCmd(install):

    def finalize_options(self) -> None:
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        return super(InstallCmd, self).finalize_options()

class Clean(distutils.command.clean.clean):
    def run(self):
        subprocess.check_call(['git', 'clean', '-fdx'], cwd=BASE_DIR, env=os.environ)

def generate_backend_py():
    backend_f = os.path.join(BASE_DIR, "torch_tpu", "tpu/backend.py")
    with open(backend_f, 'w') as f:
        if os.environ.get("CHIP_ARCH", None) == 'bm1684x':
            f.write("BACKEND='1684X'")
        elif os.environ.get("CHIP_ARCH", None) == 'sg2260':
            f.write("BACKEND='SG2260'")
        else:
            raise RuntimeError("Failed to generate backend.py")

def get_src_py_and_dst():

    ret = []
    generated_python_files = glob.glob(
        os.path.join(BASE_DIR, "torch_tpu", '**/*.py'),
        recursive=True)

    for src in generated_python_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_tpu")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    header_files = [
        "torch_tpu/csrc/*.h",
        "torch_tpu/csrc/*/*.h",
        "torch_tpu/csrc/*/*/*.h",
        "torch_tpu/csrc/*/*/*/*.h",
        "torch_tpu/csrc/*/*/*/*/*.h",
    ]
    glob_header_files = []
    for regex_pattern in header_files:
        glob_header_files += glob.glob(os.path.join(BASE_DIR, regex_pattern), recursive=True)

    for src in glob_header_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/include/torch_tpu"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_tpu")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    torch_header_files = [
        "*/*.h",
        "*/*/*.h",
        "*/*/*/*.h",
        "*/*/*/*/*.h",
        "*/*/*/*/*/*.h"
    ]
    torch_glob_header_files = []
    for regex_pattern in torch_header_files:
        torch_glob_header_files += glob.glob(os.path.join(BASE_DIR, "patch/include", regex_pattern), recursive=True)

    for src in torch_glob_header_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/include"),
            os.path.relpath(src, os.path.join(BASE_DIR, "patch/include")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))
    return ret

class PythonPackageBuild(build_py, object):
    def run(self) -> None:
        generate_backend_py()
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)

        super(PythonPackageBuild, self).finalize_options()

class EggInfoBuild(egg_info, object):
    def finalize_options(self):
        if self.distribution.package_dir:
            self.egg_base = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
            os.makedirs(self.egg_base, exist_ok=True)
        super(EggInfoBuild, self).finalize_options()

    def run(self):
        generate_backend_py()
        super().run()

class bdist_wheel(_bdist_wheel, ExtBase):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        if SOC_CROSS:
            self.plat_name = 'manylinux2014_aarch64'

    def run(self):
        self.run_command('build')
        fw_libs = glob.glob(os.path.join(BASE_DIR, 'build/firmware_*/libfirmware.so'))
        pkg_dir = self.get_package_dir()
        for fw in fw_libs:
            target = re.match('.+firmware_(\w+).+', fw).group(1)
            self.copy_file(fw, os.path.join(pkg_dir, f'lib/{target}_{"firmware" if "cmodel" in fw else "kernel_module"}.so'))
        super().run()

include_directories = [
    BASE_DIR,
    os.path.join(TPUV7_RUNTIME_PATH, "tpuv7-emulator_0.1.0", "include"),
    os.path.join(SGDNN_PATH, "include"),
    os.path.join(SGAPI_STRUCT_PATH, "include")
]
lib_directories = [
    os.path.join(BASE_DIR, f"build/{get_build_type()}/packages", "torch_tpu/lib"),
]

DEBUG = 1 #DEBUG = (os.getenv('DEBUG', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])

extra_link_args = []
extra_compile_args = [
    '-std=c++17',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type',
    '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
    '-D_GLIBCXX_USE_CXX11_ABI=0'
]

if os.environ.get("CHIP_ARCH", None) == 'sg2260':
    extra_compile_args += ["-DBACKEND_SG2260"]

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g', '-Wl,-z,now']
else:
    extra_compile_args += ['-DNDEBUG']
    extra_link_args += ['-Wl,-z,now,-s']

from setuptools.command.develop import develop as _develop
class CustomDevelop(_develop):
    def initialize_options(self):
        for ext in self.distribution.ext_modules:
            for i, v in enumerate(ext.library_dirs):
                if not os.path.abspath(v).startswith(BASE_DIR):
                    continue
                dev_path = os.path.join(BASE_DIR, 'torch_tpu/lib')
                ext.library_dirs[i] = dev_path
                break
        self.distribution.package_dir = None
        print("package_dir has been unset for the develop command.")
        _develop.initialize_options(self)

setup(
        name=os.environ.get('TORCH_TPU_PACKAGE_NAME', 'torch_tpu'),
        version=VERSION,
        author="sophgo",
        author_email="dev@sophgo.com",
        description='TPU bridge for PyTorch',
        url='https://github.com/sophgo/torch-tpu',
        packages=["torch_tpu"],
        libraries=[('torch_tpu', {'sources': list()})],
        package_dir={'': os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))},
        ext_modules=[
            CppExtension(
                'torch_tpu._C',
                sources=["torch_tpu/csrc/InitTpuBindings.cpp"],
                libraries=["torch_tpu"],
                include_dirs=include_directories,
                extra_compile_args=extra_compile_args + ['-fstack-protector-all'],
                library_dirs=lib_directories,
                extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/lib'],
            ),
        ],
        python_requires=">=3.8",
        install_requires = [],
        dependency_links = [
            "https://download.pytorch.org/whl/cpu",
        ],
        extras_require={
        },
        package_data={
            'torch_tpu': [
                '*.so', 'lib/*.so*',
            ],
        },
        cmdclass={
            'develop': CustomDevelop,
            'build_clib': CPPLibBuild,
            'build_ext': Build,
            'build_py': PythonPackageBuild,
            'bdist_wheel': bdist_wheel,
            'egg_info': EggInfoBuild,
            'clean': Clean,
            'install': InstallCmd
        })
