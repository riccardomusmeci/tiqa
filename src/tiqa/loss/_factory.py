from typing import List

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

FACTORY = {
    "mse": torch.nn.MSELoss,
}

__all__ = ["list_criteria", "create_criterion"]


def list_criteria() -> List[str]:
    """List available criteria.

    Returns:
        list: list of available criteria
    """
    return list(FACTORY.keys())


def create_criterion(name: str = "mse", **kwargs) -> _Loss:  # type: ignore
    """Create a loss criterion.

    Args:
        name (str, optional): loss criterion name. Defaults to "mse".

    Returns:
        nn.Module: loss criterion
    """
    name = name.lower()
    assert (
        name in FACTORY.keys()
    ), f"Only {list(FACTORY.keys())} criterions are supported. Change {name} to one of them."

    return FACTORY[name](**kwargs)
