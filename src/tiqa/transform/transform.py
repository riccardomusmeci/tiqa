from typing import List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import PIL
from PIL import Image


class Transform:
    """Data Augmentation for IQA.

    Args:
        train (bool): train mode
        input_size (Union[int, list, tuple]): input size
        interpolation (int, optional): interpolation. Defaults to 2.
        mean (Tuple[float, float, float], optional): normalization mean. Defaults to (0.485, 0.456, 0.406).
        std (Tuple[float, float, float], optional): normalizaton std. Defaults to (0.229, 0.224, 0.225).
        horizontal_flip_prob (float, optional): horizontal flip probability. Defaults to 0.1.
    """

    def __init__(
        self,
        train: bool,
        input_size: Union[int, list, tuple],
        interpolation: int = 2,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip_prob: float = 0.1,
    ) -> None:
        if isinstance(input_size, tuple) or isinstance(input_size, list):
            height = input_size[0]
            width = input_size[1]
        else:
            height = input_size
            width = input_size

        if train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=horizontal_flip_prob),
                    A.Resize(height=height, width=width, interpolation=interpolation, always_apply=True),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=height, width=width),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )

    def __call__(self, image: Union[np.array, PIL.Image.Image]) -> np.array:  # type: ignore
        """Apply augmentations.

        Args:
            img (Union[np.array, PIL.Image.Image]): input image

        Returns:
            np.array: transformed image
        """

        if isinstance(image, Image.Image):
            image = np.array(image)
        image = self.transform(image=image)["image"]
        return image
