import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiqa.dataset import InferenceDataset
from tiqa.io import TiqaConfiguration
from tiqa.model import create_model
from tiqa.transform import Transform
from tiqa.utils import get_device


def predict(
    config_path: Union[str, Path],
    ckpt_path: Union[str, Path],
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    to_replace: str = "model.",
) -> None:
    config = TiqaConfiguration.load(config_path=config_path)

    dataset = InferenceDataset(
        data_dir=images_dir,
        transform=Transform(train=False, **config["transform"]),
        engine=config["datamodule"]["engine"],
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

    scores_data = {"file_path": [], "IQA_score": []}  # type: ignore

    print(f"> Computing IQA scores for images at {images_dir}")
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            x, file_paths = batch
            x = x.to(device)
            out = model(x)
            scores_data["file_path"].extend(file_paths)
            scores_data["IQA_score"].extend(list(out[:, 0].cpu().numpy()))

    os.makedirs(output_dir, exist_ok=True)
    print(f"> Saving results at {os.path.join(output_dir, 'predictions.csv')}")
    scores = pd.DataFrame(scores_data)  # t
    scores.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
