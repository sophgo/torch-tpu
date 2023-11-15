from torch.random import manual_seed as torch_manual_seed, seed as torch_seed
import torch_tpu


def manual_seed(seed):
    r"""Sets the seed for generating random numbers. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed.
    """
    _seed = int(seed)
    if not torch_tpu.tpu._in_bad_fork:
        torch_tpu.tpu.manual_seed_all(_seed)

    return torch_manual_seed(_seed)


def seed():
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number. Returns a 64 bit number used to seed the RNG.
    """
    _seed = torch_seed()
    if not torch_tpu.tpu._in_bad_fork:
        torch_tpu.tpu.manual_seed_all(_seed)

    return _seed
