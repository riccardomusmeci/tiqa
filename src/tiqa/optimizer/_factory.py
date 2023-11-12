from typing import Iterable

from torch.optim import SGD, Adam, AdamW, Optimizer

FACTORY = {"sgd": SGD, "adam": Adam, "adamw": AdamW}


def create_optimizer(name: str, params: Iterable, **kwargs) -> Optimizer:  # type: ignore
    """Create an optimizer.

    Args:
        name (str): optimizer name
        params (Iterable): model parameters to optimize

    Returns:
        Optimizer: optimizer
    """
    name = name.lower()
    assert (
        name in FACTORY.keys()
    ), f"Only {list(FACTORY.keys())} optimizers are supported. Change {name} to one of them."

    return FACTORY[name](params, **kwargs)
