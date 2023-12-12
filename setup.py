import os
import sys
import subprocess
import platform
import traceback
import multiprocessing
import glob
import shutil
import re

from sysconfig import get_paths

import distutils.command.clean
from distutils.version import LooseVersion
from distutils.command.build_py import build_py

from setuptools import setup, Extension, distutils
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
THIRD_PARTY_PATH = os.path.join(BASE_DIR, "third_party")
VERSION = '2.0.1.post1'

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
    build_type = "Debug"
    # build_type = 'Release'
    # if os.getenv('DEBUG', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
    #     build_type = 'Debug'

    # if os.getenv('REL_WITH_DEB_INFO', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']:
    #     build_type = 'RelWithDebInfo'

    return build_type

def get_pytorch_dir():
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
    kwargs['include_dirs'] = temp_include_dirs

    temp_library_dirs = kwargs.get('library_dirs', [])
    temp_library_dirs.append(os.path.join(pytorch_dir, 'lib'))
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
    print(kwargs['libraries'])
    return Extension(name, sources, *args, **kwargs)

class CPPLibBuild(build_clib, object):
    def run(self):
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
            # '-DCMAKE_INSTALL_PREFIX=' + os.path.realpath(output_lib_path),
            # '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            # '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            # '-DTORCHTPU_INSTALL_LIBDIR=' + os.path.realpath(output_lib_path),
            '-DPYTHON_INCLUDE_DIR=' + get_paths()['include'],
            # '-DTORCH_VERSION=' + VERSION,
            '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir()
            ]

        build_args = ['-j', str(multiprocessing.cpu_count())]

        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        if os.environ.get("CHIP_ARCH", None) == "bm1684x":
            subprocess.check_call(['make', 'kernel_module'], cwd=build_type_dir, env=os.environ)
        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)
        generate_libs = glob.glob(os.path.join(build_type_dir, "*/*.so"), recursive=True)
        for lib in generate_libs:
            subprocess.check_call(['cp']+[lib, output_lib_path], cwd=build_type_dir, env=os.environ)

class Build(build_ext, object):

    def run(self):
        self.run_command('build_clib')
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        self.build_temp = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}"))
        self.library_dirs.append(
            os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/lib")))
        super(Build, self).run()

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
            'torch_npu/csrc/aten/RegisterCPU.cpp',
            'torch_npu/csrc/aten/RegisterNPU.cpp',
            'torch_npu/csrc/aten/RegisterAutogradNPU.cpp',
            'torch_npu/csrc/aten/RegisterUnsupprotNPU.cpp',
            'torch_npu/csrc/aten/NPUNativeFunctions.h',
            'torch_npu/csrc/aten/python_custom_functions.cpp',
            'torch_npu/utils/torch_funcs.py',
            'torch_npu/version.py',
        ]
        for remove_file in remove_files:
            file_path = os.path.join(BASE_DIR, remove_file)
            if os.path.exists(file_path):
                os.remove(file_path)

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

include_directories = [
    BASE_DIR,
]
lib_directories = [
    os.path.join(BASE_DIR, f"build/{get_build_type()}/packages", "torch_tpu/lib"),
]

DEBUG = 1 #DEBUG = (os.getenv('DEBUG', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])

extra_link_args = []
extra_compile_args = [
    '-std=c++14',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type',
    '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
    '-D_GLIBCXX_USE_CXX11_ABI=0'
]

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g', '-Wl,-z,now']
else:
    extra_compile_args += ['-DNDEBUG']
    extra_link_args += ['-Wl,-z,now,-s']

setup(
        name=os.environ.get('TORCH_TPU_PACKAGE_NAME', 'torch_tpu'),
        version=VERSION,
        description='TPU bridge for PyTorch',
        url='',
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
            'egg_info': EggInfoBuild,
            'clean': Clean,
            'install': InstallCmd
        })