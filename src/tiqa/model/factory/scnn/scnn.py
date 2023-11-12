import torch
import torch.nn as nn

from .utils import weight_init


class SCNN(nn.Module):
    """SCNN model.

    Args:
        num_classes (int): number of classes.
        init_weights (bool, optional): whether to initialize weights. Defaults to True.
    """

    def __init__(
        self,
        num_classes: int,
        init_weights: bool = True,
    ):
        super().__init__()

        self.features: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, 2, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        if init_weights:
            weight_init(self.features)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        if init_weights:
            weight_init(self.projection)

        self.num_class = num_classes
        self.classifier = nn.Linear(256, self.num_class)
        if init_weights:
            weight_init(self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """

        x = self.features(x)
        x = self.pooling(x)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out
