import pytest
import builtins
import torch
from contextlib import contextmanager

# 导入必要的torch_tpu模块
try:
    import torch_tpu

    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False


def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption(
        "--enable_profile",
        action="store_true",
        default=False,
        help="Enable TPU profiling for each test function",
    )
    parser.addoption(
        "--priority",
        action="store",
        default=None,
        help="Run only tests with specified priority (high, medium, low)",
    )


def pytest_configure(config):
    """配置 pytest，注册自定义标记"""
    config.addinivalue_line("markers", "priority_high: mark test as high priority")
    config.addinivalue_line("markers", "priority_medium: mark test as medium priority")
    config.addinivalue_line("markers", "priority_low: mark test as low priority")


def pytest_collection_modifyitems(config, items):
    """根据优先级过滤测试项"""
    priority_filter = config.getoption("--priority")

    if priority_filter:
        # 第一步：根据指定的优先级过滤测试项
        filtered_items = []

        for item in items:
            if (
                not item.get_closest_marker("priority_low")
                and not item.get_closest_marker("priority_medium")
                and not item.get_closest_marker("priority_high")
            ):
                # 没有标记的都跑
                filtered_items.append(item)
                continue

            # 第二步：检查测试项是否有对应的优先级标记
            if priority_filter == "high" and item.get_closest_marker("priority_high"):
                filtered_items.append(item)
            elif priority_filter == "medium" and item.get_closest_marker(
                "priority_medium"
            ):
                filtered_items.append(item)
            elif priority_filter == "low" and item.get_closest_marker("priority_low"):
                filtered_items.append(item)

        # 第四步：更新测试项列表
        items[:] = filtered_items


class ProfilerManager:
    """性能分析管理器，提供精确的性能分析控制"""

    def __init__(self, enable_global_profile=False, test_case=None, parametrized_items=None):
        self.enable_global_profile = enable_global_profile
        self.test_case = test_case or "unknown_test"
        self.parametrized_items = parametrized_items or {}
        self.is_profiling = False

    def enable(self, buffer_size=1024, trace_level=2):
        """手动启用性能分析"""
        if not self.is_profiling:
            try:
                torch.ops.my_ops.enable_profile(buffer_size, trace_level)
                self.is_profiling = True
                if self.enable_global_profile:
                    # 第一步：格式化参数化项信息
                    param_info = ""
                    if self.parametrized_items:
                        param_info = f" | params: {self.parametrized_items}"
                    
                    # 第二步：输出启用性能分析的详细信息
                    print(
                        f"[PROFILER] Enabled profiling for test: {self.test_case}{param_info} "
                        f"(buffer_size={buffer_size}, trace_level={trace_level})"
                    )
            except Exception as e:
                if self.enable_global_profile:
                    print(f"[PROFILER] Warning: Failed to enable profiling for {self.test_case}: {e}")

    def disable(self):
        """手动禁用性能分析"""
        if self.is_profiling:
            try:
                torch.ops.my_ops.disable_profile()
                self.is_profiling = False
                if self.enable_global_profile:
                    # 第一步：输出禁用性能分析的详细信息
                    print(f"[PROFILER] Disabled profiling for test: {self.test_case}")
            except Exception as e:
                if self.enable_global_profile:
                    print(f"[PROFILER] Warning: Failed to disable profiling for {self.test_case}: {e}")

    @contextmanager
    def profile(self, buffer_size=1024, trace_level=2):
        """上下文管理器，用于在特定代码块中启用性能分析"""
        self.enable(buffer_size, trace_level)
        try:
            yield
        finally:
            self.disable()


@pytest.fixture(scope="function")
def profiler(request):
    """
    Profiler fixture，提供精确的性能分析控制

    使用方法：
    1. 上下文管理器方式：
       with profiler.profile():
           # 需要性能分析的代码
           result = some_computation()

    2. 手动控制方式：
       profiler.enable()
       # 需要性能分析的代码
       result = some_computation()
       profiler.disable()
    """
    enable_global_profile = request.config.getoption("--enable_profile")
    
    # 第一步：获取当前测试的基本信息
    test_case = request.node.name
    
    # 第二步：提取参数化项信息
    parametrized_items = {}
    if hasattr(request.node, 'callspec'):
        # 从 callspec 中提取参数化的参数和值
        parametrized_items = dict(request.node.callspec.params)
    
    # 第三步：创建包含测试信息的 ProfilerManager
    manager = ProfilerManager(enable_global_profile, test_case, parametrized_items)

    yield manager

    # 确保在测试结束时禁用性能分析
    if manager.is_profiling:
        manager.disable()


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


@pytest.fixture(scope="function")
def setup_random_seed():
    """
    为每个测试函数设置随机种子，确保结果可重现
    """
    torch.manual_seed(1000)
    torch.set_printoptions(precision=6)


@pytest.fixture(scope="function", autouse=True)
def setup_profiling(request):
    """
    自动应用的 fixture，根据命令行参数控制性能分析

    当使用 --enable_profile 参数时：
    - 在每个测试函数执行前启用性能分析
    - 在每个测试函数执行后禁用性能分析
    """
    # 第一步：检查是否启用了 profile 选项
    enable_profile = request.config.getoption("--enable_profile")

    if enable_profile:
        try:
            # 第二步：在测试函数执行前启用性能分析
            torch.ops.my_ops.enable_profile(1024, 2)
            print(f"\n[PROFILE] Enabled profiling for test: {request.node.name}")
        except Exception as e:
            # 如果启用性能分析失败，打印警告但不影响测试执行
            print(f"\n[PROFILE] Warning: Failed to enable profiling: {e}")

    # 第三步：执行测试函数
    yield

    if enable_profile:
        try:
            # 第四步：在测试函数执行后禁用性能分析
            torch.ops.my_ops.disable_profile()
            print(f"\n[PROFILE] Disabled profiling for test: {request.node.name}")
        except Exception as e:
            # 如果禁用性能分析失败，打印警告但不影响测试执行
            print(f"\n[PROFILE] Warning: Failed to disable profiling: {e}")
