import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import IQADataset
from ..io import TiqaConfiguration
from ..model import create_model
from ..transform import DivTargetBy, Transform
from ..utils import get_device


def eval(
    ckpt_path: Union[str, Path],
    config_path: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    to_replace: str = "model.",
) -> None:
    """Predict from classification model.

    Args:
        ckpt_path (Union[str, Path]): path to checkpoint
        config_path (Union[str, Path]): path to configuration file
        images_dir (Union[str, Path]): path to images directory
        output_dir (Union[str, Path]): path to output directory
        gradcam (bool): whether to save gradcam
        layer (str): layer to use for gradcam
        gradcam_with_preds (bool): whether to save gradcam images split in folders by prediction class. Defaults to True.
        save_images (bool, optional): whether to save images. Defaults to True.
    """

    os.path.join(output_dir, "predictions.csv")

    config = TiqaConfiguration.load(config_path=config_path)

    dataset = IQADataset(
        root_dir=data_dir,
        transform=Transform(train=False, **config["transform"]),
        engine=config["datamodule"]["engine"],
        target_transform=DivTargetBy(config["datamodule"]["div_target_by"]),
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config["datamodule"]["batch_size"],
        num_workers=config["datamodule"]["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    device = get_device()
    model = create_model(model_name=config["model"]["model_name"], ckpt_path=ckpt_path, to_replace=to_replace)

    model.to(device)
    model.eval()

    print("> Running test..")
    targets = []
    scores = []
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            x, target = (el.to(device) for el in batch)
            target = target.view(len(target), 1).to(device)
            x = x.to(device)
            logits = model(x)
            scores.extend(logits[:, 0].cpu().tolist())
            targets.extend(target[:, 0].cpu().tolist())

    srcc = spearmanr(scores, targets)[0]
    plcc = pearsonr(scores, targets)[0]

    print("\n> Metrics report:")
    print(f"\t- SRCC: {srcc:.4f}")
    print(f"\t- PLCC: {plcc:.4f}")

    # # save predictions in csv
    report = {"image_path": dataset.images, "target": targets, "score": scores}
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report.csv")
    pd.DataFrame(report).to_csv(report_path, index=False)

    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write("Metrics report:\n")
        f.write(f"\t- SRCC: {srcc:.4f}\n")
        f.write(f"\t- PLCC: {plcc:.4f}\n")

    print("Eval done!")
