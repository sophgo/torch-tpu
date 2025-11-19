import subprocess
from ._versions import TPU_TRAIN_VERSION, BACKEND_VERSION
import os

_BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def _get_git_tag_desc():
    try:
        result = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--dirty", "--always"],
                cwd=_BASE_DIR,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "2.8.0.post1"


def _get_git_commit_hash():
    try:
        result = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=_BASE_DIR, stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_git_commit_short_hash():
    try:
        result = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_BASE_DIR,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_backend_version():
    from torch_tpu import lib_pwd, arch
    import re

    try:
        tpudnn = os.path.join(lib_pwd, f"libtorch_tpu.{arch}.so")
        if not os.path.isabs(tpudnn):
            tpudnn = os.path.normpath(os.path.join(lib_pwd, tpudnn))
        with open(tpudnn, "rb") as f:
            ret = re.search(b"tpu1686_revision_([0-9a-z]+)", f.read())
            if ret:
                return ret.group(1).decode("utf-8")
            else:
                return "unknown"
    except (FileNotFoundError, OSError):
        return "unknown"


def driver_version():
    try:
        with open("/proc/sgdrv/driver_date", "r") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return "unknown"


def get_tpu_train_version():
    ret = TPU_TRAIN_VERSION
    if ret is None:
        ret = _get_git_tag_desc()
    return ret


def get_backend_version():
    ret = BACKEND_VERSION
    if ret is None:
        ret = _get_backend_version()
    return ret


def versions():
    return {
        "version": get_tpu_train_version(),
        "backend_version": get_backend_version(),
        "driver_version": driver_version(),
    }


def get_commit_hash():
    return _get_git_commit_hash()


def get_short_hash():
    return _get_git_commit_short_hash()


def tpu_train_version():
    return _get_git_tag_desc()


def get_build_info():
    return f"torch_tpu {_get_git_tag_desc()} ({_get_git_commit_short_hash()})"


def show_versions():
    print(versions())


def pretty_version_info():
    return f"""
torch_tpu {get_tpu_train_version()}
backend_version: {get_backend_version()}
driver_version: {driver_version()}
"""
