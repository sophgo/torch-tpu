import os
import sys
import platform
import traceback
import subprocess
from sysconfig import get_paths
from setuptools import setup, Extension, distutils
from setuptools.command.build_clib import build_clib
from distutils.version import LooseVersion

FORMTER = "===="*10 + "\n" + "=== {}\n" + "===="*10

VERSION = '2.1.0.post1'
CHIP_ARCH           = os.environ["CHIP_ARCH"]  = "sg2260"
print(FORMTER.format(f"CHIP_ARCH={CHIP_ARCH}"))
os.environ['BASE_DIR']            = BASE_DIR           = os.path.dirname(os.path.realpath(__file__)) + "/.."
os.environ['SGDNN_PATH']          = SGDNN_PATH         = os.path.join(BASE_DIR,         "sgdnn")
os.environ['THIRD_PARTY_PATH']    = THIRD_PARTY_PATH   = os.path.join(BASE_DIR,         "third_party")
os.environ['TORCH_TPU_PATH']      = TORCH_TPU_PATH     = os.path.join(BASE_DIR,         "torch_tpu") 
os.environ['TPUDNN_PATH']         = TPUDNN_PATH        = os.path.join(THIRD_PARTY_PATH, "tpuDNN")
os.environ['SCCL_PATH']           = SCCL_PATH          = os.path.join(THIRD_PARTY_PATH, "sccl")
os.environ['TPUV7_RUNTIME_PATH']  = TPUV7_RUNTIME_PATH = os.path.join(THIRD_PARTY_PATH, "tpuv7_runtime/tpuv7-emulator_0.1.0")
os.environ['BMLIB_PATH']          = BMLIB_PATH         = os.path.join(THIRD_PARTY_PATH, "bmlib")
def get_pytorch_dir():
    try:
        import torch
        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)

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


def find_package_data(directory):
    paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 将相对于包目录的路径保存到列表中
            relative_path = os.path.relpath(os.path.join(root, filename), directory)
            paths.append(os.path.join(os.path.basename(directory), relative_path))
    return paths

def package_data():
    package_data_ = []
    package_data_.extend(find_package_data(os.path.join(TORCH_TPU_PATH, "include")))
    package_data_.extend(find_package_data(os.path.join(TORCH_TPU_PATH, "lib")))
    package_data_.extend(find_package_data(os.path.join(TORCH_TPU_PATH, "tpu")))
    package_data_.extend(find_package_data(os.path.join(TORCH_TPU_PATH, "utils")))
    package_data_.extend(find_package_data(os.path.join(TORCH_TPU_PATH, "dynamo")))
    package_data_.extend(find_package_data(os.path.join(TORCH_TPU_PATH, "demo")))
    return package_data_

# torch-tpu-python
class PYClibBuild():
    def run(self):
        self.cmake = get_cmake_command()
        cmake_args = [
            '-DPYTHON_INCLUDE_DIR='   +  get_paths()['include'],
            '-DPYTORCH_INSTALL_DIR='  +  get_pytorch_dir(),
            '-Dtpuv7_rpath='          +  "/opt/tpuv7/tpuv7-current/lib",
            ]
        install_cmd = 'install/strip'
        if CHIP_ARCH == 'sg2260':
            cmake_args.append('-DBACKEND_SG2260=ON')
        elif CHIP_ARCH == 'bm1684x':
            cmake_args.append('-DBACKEND_1684X=ON')
        build_dir = "./build"
        os.makedirs(build_dir, exist_ok=True)
        
        build_args = ['-j', str(os.cpu_count())]
        subprocess.check_call([self.cmake, '../csrc'] + cmake_args, cwd=build_dir, env=os.environ)
        subprocess.check_call(['make'] + build_args, cwd=build_dir, env=os.environ)
        subprocess.check_call(['make', install_cmd], cwd=build_dir, env=os.environ)
        os.system('cp ' + os.path.join(build_dir,"*.so") + " lib/")

## pybinding compile related
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
    return Extension(name, sources, *args, **kwargs)

include_directories = [
    os.path.join(TORCH_TPU_PATH,     "include"),
    os.path.join(SGDNN_PATH,         "include"),
    os.path.join(TPUDNN_PATH,        "include"),
    os.path.join(SCCL_PATH,          "include"),
    os.path.join(TPUV7_RUNTIME_PATH, "include"),
]

lib_directories    = [ os.path.join(TORCH_TPU_PATH, "lib") ]
extra_link_args    = [ '-Wl,-z,now,-s', '-Wl,-rpath,$ORIGIN/lib' ]
extra_compile_args = [
    '-std=c++17',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type',
    '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
    '-D_GLIBCXX_USE_CXX11_ABI=0',
    '-DNDEBUG',
    '-DBACKEND_SG2260' if os.environ.get("CHIP_ARCH", None) == 'sg2260' else '',
    '-fstack-protector-all'
]

if __name__ == "__main__":
    pyc = PYClibBuild()
    pyc.run()

    setup(
            name=os.environ.get('TORCH_TPU_PACKAGE_NAME', 'torch_tpu'),
            version=VERSION,
            author="sophgo",
            author_email="dev@sophgo.com",
            description='TPU bridge for PyTorch',
            url='https://github.com/sophgo/torch-tpu',
            packages=["torch_tpu"],
            libraries=[(f'torch_tpu.{CHIP_ARCH}',{'sources': list()})],
            package_dir={'torch_tpu': TORCH_TPU_PATH},
            ext_modules=[
                CppExtension(
                    'torch_tpu._C',
                    sources             = ["InitTpuBindings.cpp"],
                    libraries           = [f'torch_tpu_python.{CHIP_ARCH}', f'torch_tpu.{CHIP_ARCH}'],
                    include_dirs        = include_directories,
                    extra_compile_args  = extra_compile_args,
                    library_dirs        = lib_directories,
                    extra_link_args     = extra_link_args,
                )
            ],
            entry_points={
                'console_scripts': [
                    'tpu_apply_all_patch=torch_tpu.utils:apply_all_patches',
                    'tpu_revert_all_patch=torch_tpu.utils:revert_all_patches',
                    'tpu_gen_sccl_rank_table=torch_tpu.utils:gen_sccl_rank_table',
                    'tpu_show_topology=torch_tpu.utils:show_topology'
                ],
            },
            python_requires=">=3.8",
            install_requires = [],
            dependency_links = [
                "https://download.pytorch.org/whl/cpu",
            ],
            package_data={
                'torch_tpu': package_data()
            },
        )
