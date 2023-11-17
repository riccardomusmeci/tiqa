from typing import Dict, List, Optional, Union
from pathlib import Path

import os

from ..pl import IQAModelModule, IQADataModule
from ..io import TiqaConfiguration
from ..transform import Transform, DivTargetBy
from ..model import create_model
from ..loss import create_criterion
from ..optimizer import create_optimizer
from ..lr_scheduler import create_lr_scheduler
from ..pl import create_callbacks
from ..utils import now

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def train(
    config_path: Union[str, Path],
    train_data_dir: Union[str, Path],
    val_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    resume_from: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> None:
    """Train classification model.

    Args:
        config_path (Union[str, Path]): path to configuration file
        train_data_dir (Union[str, Path]): path to data train directory
        val_data_dir (Union[str, Path]): path to data validation directory
        output_dir (Union[str, Path]): path to output directory
        resume_from (Optional[Union[str, Path]], optional): path to checkpoint to resume train from. Defaults to None.
        seed (int, optional): random seed (for reproducibility). Defaults to 42.
    """

    if resume_from is not None:
        output_dir = "/".join(resume_from.split("/")[:-2])  # type: ignore
    else:
        output_dir = os.path.join(output_dir, now())
        os.makedirs(output_dir, exist_ok=True)
    print("*" * 40)
    print(f"Output directory: {output_dir}")
    print("*" * 40)

    # Loading and saving configuration
    config = TiqaConfiguration.load(config_path=config_path)
    TiqaConfiguration.save(config=config, output_path=os.path.join(output_dir, "config.yaml"))

    # Setting up datamodule, model, callbacks, logger, and trainer
    datamodule = IQADataModule(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        train_transform=Transform(train=True, **config["transform"]),
        val_transform=Transform(train=False, **config["transform"]),
        **config["datamodule"],
    )

    iqa_model = create_model(**config["model"])

    criterion = create_criterion(**config["loss"])
    optimizer = create_optimizer(params=iqa_model.parameters(), **config["optimizer"])
    lr_scheduler = create_lr_scheduler(optimizer=optimizer, **config["lr_scheduler"])

    model = IQAModelModule(
        model=iqa_model, loss=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, **config["pl_model"]
    )

    callbacks = create_callbacks(output_dir=output_dir, **config["callbacks"])
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, config["logger"]["save_dir"]),
        name=config["logger"]["name"],
    )

    trainer = Trainer(logger=logger, callbacks=callbacks, **config["trainer"])

    # Training
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from)  # type: ignore
