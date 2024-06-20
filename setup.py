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

class CPPLibBuild(build_clib, object):
    def run(self):
        release_mode = os.getenv('RELEASE_MODE') == 'ON'
        if not release_mode:
            self.build()
        else:
            lib_pwd = os.path.join(BASE_DIR, "build", get_build_type(), 'packages/torch_tpu/lib/')
            os.makedirs(os.path.join(lib_pwd, 'torch_tpu_tpuv7'), exist_ok=True)
            os.makedirs(os.path.join(lib_pwd, 'torch_tpu_bmlib'), exist_ok=True)
            # build bmlib
            os.environ['CHIP_ARCH'] = 'bm1684x'
            os.environ['MODE_PATTERN'] = 'stable'
            self.build()
            subprocess.check_call(['cp']+['libtorch_tpu.so', 'torch_tpu_bmlib/libtorch_tpu.so'], cwd=lib_pwd, env=os.environ)
            # build tpuv7_runtime
            os.environ['CHIP_ARCH'] = 'sg2260'
            os.environ['MODE_PATTERN'] = 'stable'
            self.build()
            subprocess.check_call(['cp']+['libtorch_tpu.so', 'torch_tpu_tpuv7/libtorch_tpu.so'], cwd=lib_pwd, env=os.environ)

    def build(self):
        cmake = get_cmake_command()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake

        build_dir = os.path.join(BASE_DIR, "build")
        build_type_dir = os.path.join(build_dir, get_build_type())
        output_lib_path = os.path.join(build_type_dir, "packages/torch_tpu/lib")
        os.makedirs(build_type_dir, exist_ok=True)
        os.makedirs(output_lib_path, exist_ok=True)
        self.build_lib = os.path.relpath(os.path.join(build_dir, "packages/torch_tpu"))
        self.build_temp = os.path.relpath(build_type_dir)

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DPYTHON_INCLUDE_DIR=' + python_path_dir(),
            '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir(),
            '-DBUILD_LIBTORCH=0'
            ]
        build_args = ['-j', str(8)]
        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        if os.environ.get("CHIP_ARCH", None) == "bm1684x":
            subprocess.check_call(['make', 'kernel_module'], cwd=build_type_dir, env=os.environ)
        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)

        generate_libs = glob.glob(os.path.join(build_type_dir, "*/*.so"), recursive=True)
        for lib in generate_libs:
            subprocess.check_call(['cp']+[lib, output_lib_path], cwd=build_type_dir, env=os.environ)

        if os.environ.get("MODE_PATTERN", None) != "local": #cmodel mode will no release
            subprocess.check_call(["patchelf", "--set-rpath",
                                    "$ORIGIN",
                                    os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/lib/libtorch_tpu.so"))
                                    ])
class Build(build_ext, object):

    def run(self):
        self.run_command('build_clib')
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}"))
        self.library_dirs.append(
            os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/lib")))
        super(Build, self).run()

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
        f_ignore = open('.gitignore', 'r')
        ignores = f_ignore.read()
        pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
        for wildcard in filter(None, ignores.split('\n')):
            match = pat.match(wildcard)
            if match:
                if match.group(1):
                    # Marker is found and stop reading .gitignore.
                    break
                # Ignore lines which begin with '#'.
            else:
                for filename in glob.glob(wildcard):
                    if os.path.islink(filename):
                        raise RuntimeError(f"Failed to remove path: {filename}")
                    if os.path.exists(filename):
                        try:
                            shutil.rmtree(filename, ignore_errors=True)
                        except Exception as err:
                            raise RuntimeError(f"Failed to remove path: {filename}") from err
        f_ignore.close()

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

        remove_files = [
        ]
        for remove_file in remove_files:
            file_path = os.path.join(BASE_DIR, remove_file)
            if os.path.exists(file_path):
                os.remove(file_path)

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

    generate_backend_py()
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
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(PythonPackageBuild, self).finalize_options()

class EggInfoBuild(egg_info, object):
    def finalize_options(self):
        self.egg_base = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        ret = get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(EggInfoBuild, self).finalize_options()

class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        if SOC_CROSS:
            self.plat_name = 'manylinux2014_aarch64'

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
            'build_clib': CPPLibBuild,
            'build_ext': Build,
            'build_py': PythonPackageBuild,
            'bdist_wheel': bdist_wheel,
            'egg_info': EggInfoBuild,
            'clean': Clean,
            'install': InstallCmd
        })