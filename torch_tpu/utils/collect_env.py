# derived from detectron2/utils/collect_env.py,
from contextlib import suppress

from torch.utils.collect_env import *


from torch_tpu.tpu.versions import pretty_version_info
import glob


def collect_cpu_performance():
    try:
        flags = set()
        for path in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
            with open(path, "r") as f:
                flags.add(f.read().strip())

        prefix = ""
        if len(flags) > 1 or "performance" not in flags:
            prefix = """!!!!!! run the following command to switch to performance mode !!!!!

for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
echo performance | sudo tee $cpu/cpufreq/scaling_governor
done
"""
        return f"{prefix}\n\nscaling_governor: {', '.join(flags)}"
    except (FileNotFoundError, OSError):
        raise RuntimeError("failed to collect cpu performance")




def get_env_info():
    run_lambda = run
    pip_version, pip_list_output = "N/A", "N/A"
    with suppress(Exception):
        pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        if (
            not hasattr(torch.version, "hip") or torch.version.hip is None
        ):  # cuda version
            hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"
        else:  # HIP version

            def get_version_or_na(cfg, prefix):
                _lst = [s.rsplit(None, 1)[-1] for s in cfg if prefix in s]
                return _lst[0] if _lst else "N/A"

            cfg = torch._C._show_config().split("\n")
            hip_runtime_version = get_version_or_na(cfg, "HIP Runtime")
            miopen_runtime_version = get_version_or_na(cfg, "MIOpen")
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = "N/A"
        hip_compiled_version = hip_runtime_version = miopen_runtime_version = "N/A"

    sys_version = sys.version.replace("\n", " ")

    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version="{} ({}-bit runtime)".format(
            sys_version, sys.maxsize.bit_length() + 1
        ),
        python_platform=get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=get_running_cuda_version(run_lambda),
        cuda_module_loading=get_cuda_module_loading_config(),
        nvidia_gpu_models=get_gpu_info(run_lambda),
        nvidia_driver_version=get_nvidia_driver_version(run_lambda),
        cudnn_version=get_cudnn_version(run_lambda),
        hip_compiled_version=hip_compiled_version,
        hip_runtime_version=hip_runtime_version,
        miopen_runtime_version=miopen_runtime_version,
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=get_conda_packages(run_lambda),
        os=get_os(run_lambda),
        libc_version=get_libc_version(),
        gcc_version=get_gcc_version(run_lambda),
        clang_version=get_clang_version(run_lambda),
        cmake_version=get_cmake_version(run_lambda),
        caching_allocator_config=get_cachingallocator_config(),
        is_xnnpack_available=is_xnnpack_available(),
        cpu_info=get_cpu_info(run_lambda),
    )

def get_torch_tpu_env_info():
    try:
        return pretty_str(get_env_info())
    except Exception as e:
        return f"failed to collect torch_tpu env info: {e}"

def collect_env_info():
    content = [
        get_torch_tpu_env_info(),
        "\n\n--- TPU Train environment info ---",
        pretty_version_info(),
        collect_cpu_performance(),
    ]

    return "\n".join(content)


def show_collect_env_info():
    print(collect_env_info())


if __name__ == "__main__":
    show_collect_env_info()
