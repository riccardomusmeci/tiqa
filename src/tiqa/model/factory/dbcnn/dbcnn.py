import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from ..scnn import SCNN


class DBCNN(nn.Module):
    """Deep Bilinear Convolutional Neural Network (DBCNN) model.

    Args:
        encoder (nn.Module): Encoder module.
        scnn (nn.Module): SCNN module.
        encoder_dim (int): Dimension of the encoder output.
        scnn_dim (int, optional): Dimension of the SCNN output. Defaults to 128.
        freeze_encoder (bool, optional): Whether to freeze the encoder parameters. Defaults to True.
        freeze_scnn (bool, optional): Whether to freeze the SCNN parameters. Defaults to True.
    """

    def __init__(
        self,
        encoder: nn.Module,
        scnn: nn.Module,
        encoder_dim: int,
        scnn_dim: int = 128,
        freeze_encoder: bool = True,
        freeze_scnn: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.scnn = scnn
        self.encoder_dim = encoder_dim
        self.scnn_dim = scnn_dim

        self.fc = nn.Linear(in_features=self.encoder_dim * self.scnn_dim, out_features=1)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_scnn:
            for param in self.scnn.parameters():
                param.requires_grad = False

        # init fc layer
        nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias.data, val=0)

    def unfreeze(self) -> None:
        """Unfreeze the encoder and SCNN parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.scnn.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """

        # batch size
        N = x.size()[0]
        x1 = self.encoder.forward(x)

        H = x1.size()[2]
        W = x1.size()[3]

        x2 = self.scnn.forward(x)
        H2 = x2.size()[2]
        W2 = x2.size()[3]

        if (H != H2) | (W != W2):
            x2 = F.interpolate(x2, (H, W))

        x1 = x1.view(N, self.encoder_dim, H * W)
        x2 = x2.view(N, self.scnn_dim, H * W)

        # Bilinear
        x = torch.bmm(x1, torch.transpose(x2, 1, 2)) / (H * W)
        x = x.view(N, self.encoder_dim * self.scnn_dim)
        x = torch.sqrt(x + 1e-8)
        x = torch.nn.functional.normalize(x)
        x = self.fc(x)

        return x


def dbcnn_vgg16(freeze_encoder: bool = False, freeze_scnn: bool = False) -> DBCNN:
    """Create a DBCNN model with VGG16 encoder.

    Args:
        freeze_encoder (bool, optional): If True, freeze the weights of the encoder during training. Defaults to False.
        freeze_scnn (bool, optional): If True, freeze the weights of the SCNN during training. Defaults to False.

    Returns:
        DBCNN: A DBCNN model with a VGG16 encoder and an SCNN.
    """
    vgg16 = create_model("vgg16", pretrained=True, num_classes=0)
    scnn = SCNN(num_classes=35)
    return DBCNN(
        encoder=vgg16.features,
        scnn=scnn.features,
        encoder_dim=512,
        scnn_dim=128,
        freeze_encoder=freeze_encoder,
        freeze_scnn=freeze_scnn,
    )


def dbcnn_effnetb0(freeze_encoder: bool = True, freeze_scnn: bool = True) -> DBCNN:
    effnetb0 = create_model("efficientnet_b0", pretrained=True, num_classes=0)
    scnn = SCNN(num_classes=35)
    return DBCNN(
        encoder=effnetb0,
        scnn=scnn,
        encoder_dim=1280,
        scnn_dim=128,
        freeze_encoder=freeze_encoder,
        freeze_scnn=freeze_scnn,
    )
