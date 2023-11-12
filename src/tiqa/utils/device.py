import os

import torch


def get_device() -> str:
    """Return device type of current machine.

    Returns:
        str: device type (mps, cuda, cpu)
    """
    if torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_num_workers(mode: str = "max") -> int:
    """Return number of workers for dataloader.

    Args:
        mode (str, optional): half/max. Defaults to "max".

    Returns:
        int: number of workers
    """
    if mode == "half":
        return os.cpu_count() // 2  # type: ignore
    else:
        return os.cpu_count()  # type: ignore
