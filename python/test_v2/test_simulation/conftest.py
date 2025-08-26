# auto inject torch
def pytest_configure(config):
    from torch_tpu.utils.reflection.torch_inject import inject
    import torch

    # torch.ops.my_ops.forbid_atomic_cmodel()

    inject()


def pytest_unconfigure(config):
    from torch_tpu.utils.reflection.torch_inject import restore

    restore()
