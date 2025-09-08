import pytest
import torch

try:
    import torch_tpu

    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


@pytest.fixture(scope="function")
def setup_random_seed():
    """
    为每个测试函数设置随机种子，确保结果可重现
    """
    torch.manual_seed(1000)
    torch.set_printoptions(precision=6)


# auto inject torch
def pytest_configure(config):
    from torch_tpu.utils.reflection.torch_inject import inject
    import torch

    inject()


def pytest_unconfigure(config):
    from torch_tpu.utils.reflection.torch_inject import restore

    restore()


@pytest.fixture(scope="session")
def device():
    """
    为测试提供设备参数的fixture

    根据环境自动选择可用设备：
    1. 如果TPU可用，使用 "tpu:0"
    2. 如果CUDA可用，使用 "cuda:0"
    3. 否则使用 "cpu"
    """
    if TPU_AVAILABLE:
        return "tpu:0"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        return "cpu"


@pytest.fixture(scope="session", params=["cpu", "tpu:0"])
def device_parametrized(request):
    """
    参数化的设备fixture，用于在多个设备上运行相同的测试

    使用方法：将测试函数参数从 device 改为 device_parametrized
    这样测试会在 CPU 和 TPU 上都运行一遍
    """
    device_name = request.param

    # 检查设备是否可用
    if device_name.startswith("tpu") and not TPU_AVAILABLE:
        pytest.skip(f"TPU not available, skipping test for {device_name}")
    elif device_name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"CUDA not available, skipping test for {device_name}")

    return device_name
