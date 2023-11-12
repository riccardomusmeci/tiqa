import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..io import read_rgb


class InferenceDataset(Dataset):
    """Initializes a new instance of the InferenceDataset class.

    Args:
        data_dir (Union[str, Path]): the path to the directory containing the images.
        transform (Optional[Callable], optional): a function that takes in an image and returns a transformed version of it. Defaults to None.
        engine (str, optional): the image processing engine to use. Defaults to "pil".
    """

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".gif")

    def __init__(self, data_dir: Union[str, Path], transform: Optional[Callable] = None, engine: str = "pil") -> None:
        assert os.path.exists(data_dir), f"Dataset path {data_dir} does not exist"

        self.images = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.splitext(f)[-1] in self.EXTENSIONS
        ]
        self.transform = transform
        self.engine = engine

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """Get item from dataset.

        Args:
            index (int): index

        Returns:
            Tuple[torch.Tensor, str]: image and file_path
        """
        file_path = self.images[index]
        img = read_rgb(file_path, engine=self.engine)

        if self.transform is not None:
            img = self.transform(img)

        img = torch.from_numpy(img.transpose(2, 0, 1))  # type: ignore
        return img, file_path

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            int: dataset length
        """
        return len(self.images)
