import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..io import read_rgb


class IQADataset(Dataset):
    """IQA Dataset class.

    Args:
        root_dir (Union[str, Path]): root directory of the dataset.
        transform (Optional[Callable], optional): a function/transform that takes in a sample and returns a transformed version. Defaults to None.
        engine (str, optional): image processing engine to use (cv2/pil). Defaults to "pil".
        target_transform (Optional[Callable], optional): a function/transform that takes in the target and transforms it. Defaults to None.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        engine: str = "pil",
        target_transform: Optional[Callable] = None,
    ) -> None:
        assert os.path.exists(root_dir), f"Dataset path {root_dir} does not exist"

        assert os.path.exists(
            os.path.join(root_dir, "annotations.csv")
        ), f"Dataset path {root_dir} does not contain annotations.csv"

        assert os.path.exists(
            os.path.join(root_dir, "images")
        ), f"Dataset path {root_dir} does not contain images folder"

        self.images, self.targets = self._load_samples(root_dir)
        self.transform = transform
        self.engine = engine
        self.target_transform = target_transform

    def _load_samples(self, root_dir: Union[str, Path]) -> Tuple[List[str], List[float]]:
        """Load samples from dataset.

        Args:
            root_dir (Union[str, Path]): dataset root directory

        Returns:
            Tuple[List[str], List[float]]: list of images and targets
        """

        images = []
        targets = []
        fail = []

        annotations = pd.read_csv(os.path.join(root_dir, "annotations.csv"))
        for _i, row in annotations.iterrows():
            if os.path.exists(os.path.join(root_dir, "images", row["image_name"])):
                images.append(os.path.join(root_dir, "images", row["image_name"]))
                targets.append(row["MOS"])
            else:
                fail.append(row["image_name"])
        if len(fail) > 0:
            print(f"> [WARNING]{len(fail)} images not found in dataset")
        return images, targets

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        """Get item from dataset.

        Args:
            index (int): index

        Returns:
            Tuple[torch.Tensor, float]: image and target
        """
        file_path = self.images[index]
        target = self.targets[index]
        img = read_rgb(file_path, engine=self.engine)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = torch.from_numpy(img.transpose(2, 0, 1))  # type: ignore
        target = torch.tensor(target, dtype=torch.float32)  # type: ignore
        return img, target

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: dataset length
        """
        return len(self.images)
