import os
import sys
import subprocess
import platform
import traceback
import glob
import re
import site
import sysconfig
from pathlib import Path

from sysconfig import get_paths

import distutils.command.clean
from distutils.version import LooseVersion

from setuptools import setup, Extension, distutils, Command
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SGAPI_STRUCT_PATH = os.path.join(BASE_DIR, "common")
THIRD_PARTY_PATH = os.path.join(BASE_DIR, "third_party")
SCCL_PATH = os.path.join(THIRD_PARTY_PATH, "sccl")
TPUV7_RUNTIME_PATH = os.path.join(THIRD_PARTY_PATH, "tpuv7_runtime")
BMLIB_PATH = os.path.join(THIRD_PARTY_PATH, "bmlib")

def is_building_wheel():
    args = set(sys.argv[1:])
    if 'bdist_wheel' in args or 'bdist' in args:
        return True
    return False
is_bdist_wheel = is_building_wheel()

def get_git_tag_desc():
    try:
        result = subprocess.check_output(
            ['git', 'describe', '--tags', '--dirty', '--always'],
            cwd=BASE_DIR,
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return '2.1.0.post1'

def get_git_commit_hash():
    try:
        result = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=BASE_DIR,
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def get_git_commit_short_hash():
    try:
        result = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=BASE_DIR,
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

VERSION = '2.1.0.post1'
GIT_VERSION = get_git_tag_desc()
SOC_CROSS = os.environ.get("SOC_CROSS_MODE", None)
SOC_MODE = os.environ.get("SOC_MODE", None)
SOC_CROSS = True if SOC_CROSS == "ON" else False
CROSS_TOOLCHAINS= os.environ.get("CROSS_TOOLCHAINS", None)
PLATFORM=''
if SOC_CROSS:
    if os.environ["CHIP_ARCH"] == "bm1684x":
        os.environ["CC"] = f"{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc"
        os.environ["CXX"] = f"{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"
        PLATFORM='-aarch64'
    else:
        os.environ["CC"] = f"{CROSS_TOOLCHAINS}/riscv64-linux-x86_64/bin/riscv64-unknown-linux-gnu-gcc"
        os.environ["CXX"] = f"{CROSS_TOOLCHAINS}/riscv64-linux-x86_64/bin/riscv64-unknown-linux-gnu-g++"
        PLATFORM='-riscv64'
if SOC_MODE:
    PLATFORM='-riscv64'

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
        return os.path.join(CROSS_TOOLCHAINS, "pythons/Python-3.8.2/python_3.8.2/include/python3.8/")
    return get_paths()['include']

def get_pytorch_dir():
    if SOC_CROSS:
        print(os.path.join(CROSS_TOOLCHAINS, "torchwhl-2.1.0-cp311/torch"))
        return os.path.join(CROSS_TOOLCHAINS, "torchwhl-2.1.0-cp311/torch")
    torch_dir_env = os.environ.get("PYTORCH_INSTALL_DIR")
    if torch_dir_env and os.path.exists(torch_dir_env):
        return torch_dir_env
    try:
        import torch
        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)

def symlink(src, dst):
    try:
        link_val = os.readlink(dst)
        if link_val == src:
            return
        else:
            os.unlink(dst)
    except FileNotFoundError:
        pass

    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass
dev_path = os.path.join(BASE_DIR, 'torch_tpu/lib')

def CppExtension(name, sources, *args, **kwargs):
    r'''
    Creates a :class:`setuptools.Extension` for C++.
    '''
    pytorch_dir = get_pytorch_dir()
    temp_include_dirs = kwargs.get('include_dirs', [])
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include'))
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include/torch/csrc/api/include'))
    if SOC_CROSS:
        temp_include_dirs.append(os.path.join(CROSS_TOOLCHAINS, "Python-3.8.2/python_3.8.2/include/python3.8/"))
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
        package_dir = os.path.join(package_dir.get(''), 'torch_tpu') if is_bdist_wheel \
            else os.path.join(BASE_DIR, 'torch_tpu')
        package_dir = os.path.abspath(package_dir)
        return package_dir

class CPPLibBuild(build_clib, ExtBase, object):
    def run(self):
        release_mode = os.getenv('RELEASE_MODE') == 'ON'
        if not release_mode:
            self.build()
            if not is_bdist_wheel:
                arch = os.environ["CHIP_ARCH"]
                symlink(
                    os.path.join(BASE_DIR, f'third_party/tpuDNN/{arch}_lib/libtpudnn.so'),
                    os.path.join(dev_path, f'libtpudnn.{arch}.so'))
                symlink(
                    os.path.join(BASE_DIR, f'third_party/runtime_api/lib_{arch}/libtpurt.so'),
                    os.path.join(dev_path, f'libtpurt.{arch}.so'))
                symlink(
                    os.path.join(BASE_DIR, f'third_party/sccl/lib/libsccl.so'),
                    os.path.join(dev_path, f'libsccl.so'))
        else:
            # build bmlib
            os.environ['CHIP_ARCH'] = 'bm1684x'
            self.build()
            os.environ['CHIP_ARCH'] = 'bm1686'
            self.build()
            # build tpuv7_runtime
            os.environ['CHIP_ARCH'] = 'sg2260'
            self.build()

            os.environ['CHIP_ARCH'] = 'sg2260e'
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
        args = [self.cmake, f'-DCHIP_ARCH={arch}', src_dir]
        extra_cmake_opts = os.environ.get('EXTRA_CONFIG')
        if extra_cmake_opts:
            args += extra_cmake_opts.split()
        def build_firmware():
            os.makedirs(build_dir, exist_ok=True)
            subprocess.check_call(args, cwd=build_dir, env=os.environ)
            subprocess.check_call(['make', '-j'], cwd=build_dir, env=os.environ)

        build_fw = True
        if arch == 'bm1684x':
            tc_path = os.environ.get('ARM_TOOLCHAIN')
        else:
            tc_path = os.environ.get('RISCV_TOOLCHAIN')
        if not tc_path or not os.path.exists(tc_path):
            build_fw = False

        if build_fw:
            build_firmware()
            fw_path = os.path.join(build_dir, 'libfirmware.so')

        build_dir += '_cmodel'
        if not build_fw:
            fw_path = os.path.join(build_dir, 'libfirmware.so')
        if not SOC_CROSS or arch == 'sg2260' or arch == 'sg2260e':
            args.append('-DUSING_CMODEL=ON')
        build_firmware()

        package_dir = self.get_package_dir()
        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DPYTHON_INCLUDE_DIR=' + python_path_dir(),
            '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir(),
            '-DBUILD_LIBTORCH=0',
            '-DHOSTCCL=OFF',
            f'-DKERNEL_MODULE_PATH={fw_path}',
            f'-DCMAKE_INSTALL_PREFIX={package_dir}'
            ]
        install_cmd = 'install/strip'
        if os.getenv('TPUTRAIN_DEBUG'):
            install_cmd = 'install'
            cmake_args.append('-DDEBUG=ON')
        if arch == 'sg2260' or arch == 'sg2260e':
            cmake_args.append('-DBACKEND_SG2260=ON')
        elif arch == 'bm1684x':
            cmake_args.append('-DBACKEND_1684X=ON')
        elif arch == 'bm1686':
            cmake_args.append('-DBACKEND_1686=ON')
        build_dir = os.path.join(BASE_DIR, 'build/torch-tpu')
        os.makedirs(build_dir, exist_ok=True)
        build_args = ['-j', str(os.cpu_count())]
        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ)
        subprocess.check_call(['make'] + build_args, cwd=build_dir, env=os.environ)
        subprocess.check_call(['make', install_cmd], cwd=build_dir, env=os.environ)

class Build(build_ext, ExtBase, object):

    def run(self):
        self.run_command('build_clib')
        package_dir = self.get_package_dir()
        sym_fnc = os.path.join(package_dir, 'lib/libtorch_tpu.so')

        if not os.path.exists(sym_fnc):
            lib_fn = glob.glob(os.path.join(package_dir, 'lib/*libtorch_tpu*.so'))[0]
            os.symlink(lib_fn, sym_fnc)

        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        self.library_dirs.append(
            os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/lib")))
        super(Build, self).run()

        if not is_bdist_wheel:
            built_exts = glob.glob(os.path.join(self.build_lib, 'torch_tpu', '_C*.so'))
            for so in built_exts:
                dst = os.path.join(package_dir, os.path.basename(so))
                if os.path.abspath(so) != os.path.abspath(dst):
                    import shutil
                    shutil.copy2(so, dst)

        os.unlink(sym_fnc)

    def finalize_options(self):
        build_ext.finalize_options(self)
        if SOC_CROSS:
            if os.environ["CHIP_ARCH"] == "bm1684x":
                self.plat_name = "aarch64"
            else:
                self.plat_name = "riscv64"

        if not is_bdist_wheel:
            dev_lib_dir = os.path.join(BASE_DIR, 'torch_tpu', 'lib')
            for ext in self.extensions:
                new_lib_dirs = []
                for v in getattr(ext, 'library_dirs', []):
                    if os.path.abspath(v).startswith(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages")):
                        new_lib_dirs.append(dev_lib_dir)
                    else:
                        new_lib_dirs.append(v)
                ext.library_dirs = new_lib_dirs
            self.library_dirs = [dev_lib_dir if os.path.abspath(p).startswith(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages")) else p
                                for p in getattr(self, 'library_dirs', [])]

    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        if 'x86_64-linux-gnu' in filename and SOC_CROSS:
            if os.environ["CHIP_ARCH"] == "bm1684x":
                return filename.replace('x86_64-linux-gnu', 'aarch64-linux-gnu')
            else:
                return filename.replace('x86_64-linux-gnu', 'riscv64-linux-gnu')
        return filename

class InstallCmd(install):

    def finalize_options(self) -> None:
        self.build_lib = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
        return super(InstallCmd, self).finalize_options()

class Clean(distutils.command.clean.clean):
    def run(self):
        if not os.environ.get("TORCH_TPU_NONINTERACTIVE"):
            input('\nsetup.py clean will run git clean -fdx, \033[1;31mTHIS IS DANGEROUS!\033[0m\nMake sure you known what you are doing.\n\nOtherwise Ctrl-C to exit NOW!\n')
        subprocess.check_call(['git', 'clean', '-fdx'], cwd=BASE_DIR, env=os.environ)

def walk_dir(dir):
    res = []
    for root, _, files in os.walk(dir):
        for file in files:
            res.append(os.path.join(root, file))
    return res

def get_src_py_and_dst():

    ret = []
    generated_python_files = glob.glob(
        os.path.join(BASE_DIR, "torch_tpu", '**/*.py'),
        recursive=True)
    demo_files = walk_dir(os.path.join(BASE_DIR, "torch_tpu/demo"))
    generated_python_files = list(set(generated_python_files) | set(demo_files))
    for src in generated_python_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_tpu")))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    tool_files = walk_dir(os.path.join(BASE_DIR, "tools"))
    for src in tool_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu"),
            os.path.relpath(src, BASE_DIR))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    header_files = [
        "torch_tpu/csrc/**/*.h", "torch_tpu/csrc/**/**/*.hpp"
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

    other_header_files = [
        "third_party/bmlib/include/*.h",
        "third_party/bmruntime/include/*.h",
    ]

    other_glob_header_files = []
    for regex_pattern in other_header_files:
        other_glob_header_files += glob.glob(os.path.join(BASE_DIR, regex_pattern), recursive=True)

    for src in other_glob_header_files:
        dst = os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/include")
        os.makedirs(dst, exist_ok=True)
        ret.append((src, os.path.join(dst, os.path.basename(src))))

    torch_header_files = [
        "**/*.h",
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

        # 在编译时生成 versions.py 文件，包含编译时的 git commit hash
        self.generate_versions_file()

        super(PythonPackageBuild, self).finalize_options()

    def extract_backend_version(self):
        """从 tpudnn.so 文件中提取后端版本信息"""
        import re

        try:
            # 搜索构建目录中的 tpudnn.so 文件
            lib_dir = os.path.join(BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/lib")

            # 查找所有包含 tpudnn.so 的文件
            tpudnn_files = []
            if os.path.exists(lib_dir):
                for file in os.listdir(lib_dir):
                    if 'tpudnn' in file and file.endswith('.so'):
                        tpudnn_files.append(os.path.join(lib_dir, file))

            # 如果构建目录中没有找到，尝试从源目录查找
            if not tpudnn_files:
                third_party_lib_dirs = [
                    os.path.join(BASE_DIR, 'third_party/tpuDNN/bm1684x_lib'),
                    os.path.join(BASE_DIR, 'third_party/tpuDNN/sg2260_lib'),
                ]

                for lib_dir in third_party_lib_dirs:
                    if os.path.exists(lib_dir):
                        for root, dirs, files in os.walk(lib_dir):
                            for file in files:
                                if 'tpudnn' in file and file.endswith('.so'):
                                    tpudnn_files.append(os.path.join(root, file))

            # 在找到的文件中搜索版本信息
            pattern = re.compile(rb'tpu1686_revision_([0-9a-z]+)')

            for so_file in tpudnn_files:
                try:
                    with open(so_file, 'rb') as f:
                        content = f.read()
                        match = pattern.search(content)
                        if match:
                            backend_version = match.group(1).decode('ascii')
                            print(f"Found backend version {backend_version} in {so_file}")
                            return backend_version
                except (IOError, OSError) as e:
                    print(f"Failed to read {so_file}: {e}")
                    continue

            print("No backend version found in tpudnn.so files")
            return "unknown"

        except Exception as e:
            print(f"Error extracting backend version: {e}")
            return "unknown"

    def generate_versions_file(self):
        from datetime import datetime
        build_version = get_git_tag_desc()
        backend_version = self.extract_backend_version()

        versions_content = f'''# Auto generated by setup.py at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

TPU_TRAIN_VERSION = "{build_version}"
BACKEND_VERSION = "{backend_version}"

'''

        versions_dst = os.path.join(
            BASE_DIR, f"build/{get_build_type()}/packages/torch_tpu/tpu/_versions.py"
        )

        os.makedirs(os.path.dirname(versions_dst), exist_ok=True)

        with open(versions_dst, 'w', encoding='utf-8') as f:
            f.write(versions_content)


class EggInfoBuild(egg_info, object):
    def finalize_options(self):
        if self.distribution.package_dir:
            self.egg_base = os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))
            os.makedirs(self.egg_base, exist_ok=True)
        super(EggInfoBuild, self).finalize_options()

    def run(self):
        super().run()

class bdist_wheel(_bdist_wheel, ExtBase):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        if SOC_CROSS:
            if os.environ["CHIP_ARCH"] == "bm1684x":
                self.plat_name = 'manylinux2014_aarch64'
            else:
                self.plat_name = 'linux_riscv64'
    def copy_file(self, src, dst):
        #check file
        def is_lfs_pointer(file_path):
            with open(file_path, 'r', errors="ignore") as f:
                header = [f.readline() for _ in range(5)]

            # 判断是否包含 Git LFS 指针文件的特征字符串
            if header and header[0].startswith("version https://git-lfs.github.com/spec"):
                print(f"[ERROR] {file_path} is a ELF HEADER FILE. \
                      \n[ERROR] Please pull with \"git lfs pull --include '*' --exclude ''\" ")
                return True
            return False
        if is_lfs_pointer(src):
            print("[ERROR] The third-party library was not pulled down !!!")
            sys.exit(0)
        super().copy_file(src, dst)

    def run(self):
        self.run_command('build')
        fw_libs = glob.glob(os.path.join(BASE_DIR, 'build/firmware_*cmodel/libfirmware.so'))
        pkg_dir = self.get_package_dir()
        for fw in fw_libs:
            target = re.match(r'.+firmware_(\w+).+', fw).group(1)
            self.copy_file(fw, os.path.join(pkg_dir, f'lib/{target}_{"firmware" if "cmodel" in fw else "kernel_module"}.so'))

        tpuDNN_libs = glob.glob(os.path.join(BASE_DIR, 'third_party/tpuDNN/*_lib/*.so'))
        for lib in tpuDNN_libs:
            target = re.match(r'.+tpuDNN/(\w+)_lib.+', lib).group(1)
            if 'tpudnn' in lib:
                if 'bm1684x' in lib:
                    if SOC_CROSS:
                        lib = os.path.join(BASE_DIR, 'third_party/tpuDNN/bm1684x_lib/arm/libtpudnn.so')
                    self.copy_file(lib, os.path.join(pkg_dir, f'lib/libtpudnn.{target}.so'))
                elif 'bm1686' in lib:
                    if SOC_CROSS:
                        lib = os.path.join(BASE_DIR, 'third_party/tpuDNN/bm1686_lib/arm/libtpudnn.so')
                    self.copy_file(lib, os.path.join(pkg_dir, f'lib/libtpudnn.{target}.so'))
                else:
                    if 'riscv' in lib and (SOC_CROSS or SOC_MODE):
                        self.copy_file(lib, os.path.join(pkg_dir, f'lib/libtpudnn.{target}.so'))
                    elif 'riscv' not in lib and not (SOC_CROSS or SOC_MODE):
                        self.copy_file(lib, os.path.join(pkg_dir, f'lib/libtpudnn.{target}.so'))

        chip_arch = os.environ.get('CHIP_ARCH')
        sccl_libs = glob.glob(os.path.join(BASE_DIR, f'third_party/sccl/lib/libsccl{PLATFORM}.so'))
        for lib in sccl_libs:
            self.copy_file(lib, os.path.join(pkg_dir, f'lib/libsccl.so'))

        # runtime libs
        runtime_libs = glob.glob(os.path.join(BASE_DIR, f'third_party/runtime_api/lib_*/libtpurt{PLATFORM}.so'))
        for lib in runtime_libs:
            arch = os.path.basename(os.path.dirname(lib)).replace('lib_', '')
            self.copy_file(lib, os.path.join(pkg_dir, f'lib/libtpurt.{arch}.so'))

        # include libraries just cmodel, for inst-cache use.
        # tpuv7-emulator_0.1.0 is the cmodel version of runtime, no device version contained.
        base_fw_libs = glob.glob(os.path.join(TPUV7_RUNTIME_PATH, f'tpuv7-emulator_0.1.0/lib/libtpu*_emulator.so')) if not (SOC_CROSS or SOC_MODE) else \
                       glob.glob(os.path.join(TPUV7_RUNTIME_PATH, f'tpuv7-emulator_0.1.0/lib/libtpu*_emulator-riscv.so'))
        for lib in base_fw_libs:
            filename = os.path.basename(lib)
            if filename.endswith('_emulator-riscv.so'):
                filename = filename.replace('_emulator-riscv.so', '_emulator.so')
            self.copy_file(lib, os.path.join(pkg_dir, f'lib/{filename}'))
        dnn_libs = glob.glob(os.path.join(TPUV7_RUNTIME_PATH, 'tpuv7-emulator_0.1.0/lib/libdnnl.so.3'))
        for lib in dnn_libs:
            target = os.path.join(pkg_dir, f'lib/', os.path.basename(lib))
            self.copy_file(lib, target)

        super().run()

include_directories = [
    BASE_DIR,
    os.path.join(BASE_DIR, 'third_party/tpuDNN/include'),
    os.path.join(BASE_DIR, 'third_party/runtime_api/include'),
    os.path.join(SCCL_PATH, 'include'),
    os.path.join(TPUV7_RUNTIME_PATH, "tpuv7-emulator_0.1.0", "include"),
    os.path.join(SGAPI_STRUCT_PATH, "include"),
    os.path.join(BMLIB_PATH, "include"),
]
if SOC_CROSS:
    if os.environ["CHIP_ARCH"] == "sg2260":
        include_directories.append(os.path.join(CROSS_TOOLCHAINS, "riscv64-linux-x86_64/python3.10"))
    elif os.environ["CHIP_ARCH"] in ["bm1684x", "bm1686"]:
        include_directories.append(os.path.join(CROSS_TOOLCHAINS, "Python-3.8.2/python_3.8.2/include/python3.8"))
lib_directories = [
    os.path.join(BASE_DIR, f"build/{get_build_type()}/packages", "torch_tpu/lib"),
] if is_bdist_wheel else [os.path.join(BASE_DIR, "torch_tpu", "lib")]

abi_flag = '1' if os.getenv('TORCH_CXX11_ABI', '0').upper() in ['1','TRUE','ON','YES','Y'] else '0'
print("D_GLIBCXX_USE_CXX11_ABI=", abi_flag)
extra_link_args = []
extra_compile_args = [
    '-std=c++17',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type',
    '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
    f'-D_GLIBCXX_USE_CXX11_ABI={abi_flag}'
]

DEBUG = os.getenv('TPUTRAIN_DEBUG', default='0').upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
if os.environ.get("CHIP_ARCH", None) == 'sg2260' or os.environ.get("CHIP_ARCH", None) == 'sg2260e':
    extra_compile_args += ["-DBACKEND_SG2260"]

if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g', '-Wl,-z,now']
else:
    extra_compile_args += ['-DNDEBUG', '-O3', '-g0']
    extra_link_args += ['-Wl,-z,now,-s,-O3']

setup(
        name=os.environ.get('TORCH_TPU_PACKAGE_NAME', 'torch_tpu'),
        version=VERSION,
        author="sophgo",
        author_email="dev@sophgo.com",
        description='TPU bridge for PyTorch',
        url='https://github.com/sophgo/torch-tpu',
        packages=["torch_tpu"],
        libraries=[('torch_tpu', {'sources': list()})],
        package_dir={'': os.path.relpath(os.path.join(BASE_DIR, f"build/{get_build_type()}/packages"))} if is_building_wheel() else None,
        ext_modules=[
            CppExtension(
                'torch_tpu._C',
                sources=["torch_tpu/csrc/InitTpuBindings.cpp"],
                include_dirs=include_directories,
                extra_compile_args=extra_compile_args + ['-fstack-protector-all'],
                library_dirs=lib_directories,
                extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/lib'],
            )
        ],
        entry_points={
            'console_scripts': [
                'tpu_apply_all_patch=torch_tpu.utils:apply_all_patches',
                'tpu_revert_all_patch=torch_tpu.utils:revert_all_patches',
                'tpu_gen_sccl_rank_table=torch_tpu.utils:gen_sccl_rank_table',
                'tpu_show_topology=torch_tpu.utils:show_topology',
                'tpu_version=torch_tpu.tpu.versions:show_versions',
                'tpu_collect_env=torch_tpu.utils.collect_env:show_collect_env_info',
            ],
        },
        python_requires=">=3.8",
        install_requires = [],
        dependency_links = [
            "https://download.pytorch.org/whl/cpu",
        ],
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