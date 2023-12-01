import timm
import torch
import torch.nn as nn


class ReIQA(nn.Module):
    """Re-IQA model.

    Args:
        encoder (nn.Module): the encoder module.
        encoder_dim (int): the dimension of the encoder output.
        out_dim (int): the dimension of the output.
        freeze_encoder (bool, optional): whether to freeze the encoder parameters. Defaults to True.
        freeze_fc (bool, optional): whether to freeze the fully connected layers parameters. Defaults to True.
        dropout (float, optional): the dropout rate. Defaults to 0.2.
        bias (bool, optional): whether to include bias in linear layers. Defaults to False.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        out_dim: int,
        freeze_encoder: bool = True,
        freeze_fc: bool = True,
        dropout: float = 0.2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(encoder_dim, encoder_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(encoder_dim, out_dim, bias=True)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if freeze_fc:
            for param in self.fc1.parameters():
                param.requires_grad = False
            for param in self.fc2.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_features=out_dim, out_features=1, bias=bias)

    def unfreeze(self, encoder: bool = True, fc: bool = True) -> None:
        """Unfreeze parameters.

        Args:
            encoder (bool, optional): if True, unfreeze encoder parameters. Defaults to True.
            fc (bool, optional): if True, unfreeze fully connected layers parameters. Defaults to True.
        """
        if encoder:
            for param in self.encoder.parameters():
                param.requires_grad = True
        if fc:
            for param in self.fc1.parameters():
                param.requires_grad = True
            for param in self.fc2.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): the input tensor.

        Returns:
            torch.Tensor: the output tensor
        """
        x = self.encoder.forward(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.head(x)
        return x


def reiqa_resnet50(
    pretrained: bool = True,
    freeze_encoder: bool = True,
    freeze_fc: bool = True,
    dropout: float = 0.2,
    bias: bool = True,
) -> ReIQA:
    encoder = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
    model = ReIQA(
        encoder=encoder,
        encoder_dim=2048,
        out_dim=128,
        freeze_encoder=freeze_encoder,
        freeze_fc=freeze_fc,
        dropout=dropout,
        bias=True,
    )
    return model
