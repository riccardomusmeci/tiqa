import os
import os.path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
import torch.utils.data as data

from ..io import read_rgb


def getFileName(path: str, suffix: str) -> List[str]:
    """Get the list of filenames with a suffix."""
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


class cropInsulator2kFolder(data.Dataset):
    """Data Loader class to load images from cropInsulator2k dataset."""

    def __init__(
        self,
        root: str,
        index: Optional[List[int]] = None,
        split: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Class definition.

        Args:
            root: Root directory path.
            loader: A function to load an image given its path.
            index: List of images indices to consider (can be None if split is
                provided).
            split: split to use for pre-splitted datasets. either 'train' or 'test'
                (can be None if index is provided).
            transform: A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform: A function/transform that takes in the
                target and transforms it.
        """

        if index is None and split is None:
            raise ValueError("Either index or split must be set.")
        if split and split not in ["test", "train"]:
            raise ValueError('split must be either "train" or "test".')

        self.root = root
        # self.input_size = input_size
        self.imgs = pd.read_csv(os.path.join(self.root, "cropIsolator2k.csv"))
        self.imgpaths = []
        self.labels = []
        for i, row in self.imgs.iterrows():
            if split and split in row["set"] or index and i in index:
                self.imgpaths.append(os.path.join(self.root, "images", row["image_name"]))
                self.labels.append(row["MOS"])

        self.samples = []
        for imgpath, label in zip(self.imgpaths, self.labels):
            self.samples.append((imgpath, label))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index: Index

        Returns:
            (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = read_rgb(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        length = len(self.samples)
        return length
