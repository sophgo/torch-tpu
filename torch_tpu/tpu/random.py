import contextlib
import warnings
import torch
import torch_tpu
from .utils import _lazy_init, _lazy_call, device_count, current_device


__all__ = ['get_rng_state', 'set_rng_state',
           'get_rng_state_all', 'set_rng_state_all',
           'manual_seed', 'manual_seed_all',
           'seed', 'seed_all', 'initial_seed']


def get_rng_state(device='tpu'):
    r"""Returns the random number generator state of the specified TPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'tpu'`` (i.e., ``torch.device('tpu')``, the current TPU device).

    .. warning::
        This function eagerly initializes TPU.
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('tpu', device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_tpu.tpu.default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all():
    r"""Returns a list of ByteTensor representing the random number states of all devices."""

    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results


def set_rng_state(new_state, device='tpu'):
    r"""Sets the random number generator state of the specified TPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'tpu'`` (i.e., ``torch.device('tpu')``, the current TPU device).
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('tpu', device)

    def cb():
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch_tpu.tpu.default_generators[idx]
        default_generator.set_state(new_state_copy)

    _lazy_call(cb)


def set_rng_state_all(new_states):
    r"""Sets the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed):
    r"""Sets the seed for generating random numbers for the current TPU.
    It's safe to call this function if TPU is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-TPU model, this function is insufficient
        to get determinism.  To seed all TPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = torch_tpu.tpu.default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb)


def manual_seed_all(seed):
    r"""Sets the seed for generating random numbers on all TPUs.
    It's safe to call this function if TPU is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = torch_tpu.tpu.default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb)


def seed():
    r"""Sets the seed for generating random numbers to a random number for the current TPU.
    It's safe to call this function if TPU is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-TPU model, this function will only initialize
        the seed on one TPU.  To initialize all TPUs, use :func:`seed_all`.
    """

    def cb():
        idx = current_device()
        default_generator = torch_tpu.tpu.default_generators[idx]
        default_generator.seed()

    _lazy_call(cb)


def seed_all():
    r"""Sets the seed for generating random numbers to a random number on all TPUs.
    It's safe to call this function if TPU is not available; in that
    case, it is silently ignored.
    """

    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch_tpu.tpu.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed():
    r"""Returns the current random seed of the current TPU.

    .. warning::
        This function eagerly initializes TPU.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch_tpu.tpu.default_generators[idx]
    return default_generator.initial_seed()


_fork_rng_warned_already = False


@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, _caller="fork_rng", _devices_kw="devices"):
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.
    Args:
        devices (iterable of TPU IDs): TPU devices for which to fork
            the RNG.  CPU RNG state is always forked.  By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
    """

    global _fork_rng_warned_already

    # Internal arguments:
    #   _caller: the function which called fork_rng, which the user used
    #   _devices_kw: the devices keyword of _caller

    if not enabled:
        yield
        return

    if devices is None:
        num_devices = torch.tpu.device_count()
        if num_devices > 1 and not _fork_rng_warned_already:
            warnings.warn(
                ("TPU reports that you have {num_devices} available devices, and you "
                 "have used {caller} without explicitly specifying which devices are being used. "
                 "For safety, we initialize *every* TPU device by default, which "
                 "can be quite slow if you have a lot of GPUs.  If you know that you are only "
                 "making use of a few TPU devices, set the environment variable CUDA_VISIBLE_DEVICES "
                 "or the '{devices_kw}' keyword argument of {caller} with the set of devices "
                 "you are actually using.  For example, if you are using CPU only, "
                 "set CUDA_VISIBLE_DEVICES= or devices=[]; if you are using "
                 "GPU 0 only, set CUDA_VISIBLE_DEVICES=0 or devices=[0].  To initialize "
                 "all devices and suppress this warning, set the '{devices_kw}' keyword argument "
                 "to `range(torch.tpu.device_count())`."
                 ).format(num_devices=num_devices, caller=_caller, devices_kw=_devices_kw))
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = torch.get_rng_state()
    gpu_rng_states = []
    for device in devices:
        gpu_rng_states.append(torch.tpu.get_rng_state(device))

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        for device, gpu_rng_state in zip(devices, gpu_rng_states):
            torch.tpu.set_rng_state(gpu_rng_state, device)
