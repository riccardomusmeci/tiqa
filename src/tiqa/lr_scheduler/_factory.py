from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

FACTORY = {
    "cosine": CosineAnnealingLR,
    "cosine_restarts": CosineAnnealingWarmRestarts,
    "linear": LinearLR,
    "step": StepLR,
    "reduce_on_plateau": ReduceLROnPlateau,
}


def create_lr_scheduler(optimizer: Optimizer, name: str, **kwargs) -> _LRScheduler:  # type: ignore
    name = name.lower()
    assert (
        name in FACTORY.keys()
    ), f"Only {list(FACTORY.keys())} lr_schedulers are supported. Change {name} to one of them."
    return FACTORY[name](optimizer, **kwargs)
