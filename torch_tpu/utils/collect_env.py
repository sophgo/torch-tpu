# derived from detectron2/utils/collect_env.py,

from torch.utils.collect_env import get_pretty_env_info


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


def collect_env_info():
    content = [
        get_pretty_env_info(),
        "\n\n--- TPU Train environment info ---",
        pretty_version_info(),
        collect_cpu_performance(),
    ]

    return "\n".join(content)


def show_collect_env_info():
    print(collect_env_info())


if __name__ == "__main__":
    show_collect_env_info()
