import os

import torch
from path import DATASET_DIR

from tiqa.dataset import InferenceDataset
from tiqa.loss import create_criterion
from tiqa.model import create_model
from tiqa.transform import Transform

torch.manual_seed(42)


def test_model() -> None:
    model = create_model(model_name="dbcnn_vgg16", pretrained=False, ckpt_path=None)
    inference_dataset = InferenceDataset(
        data_dir=os.path.join(DATASET_DIR, "images"), transform=Transform(train=False, input_size=224)
    )
    x, _ = inference_dataset[0]
    x = x.unsqueeze(dim=0)
    out = model(x).squeeze(dim=0)
    assert out.shape == torch.Size([1])


def test_loss() -> None:
    loss = create_criterion("mse")
    model = create_model(model_name="dbcnn_vgg16", pretrained=False, ckpt_path=None)
    x = torch.rand((1, 3, 224, 224))
    scores = model(x)
    target = torch.randint(1, (1,), dtype=torch.int64)
    loss_value = loss(scores, target)
    assert loss_value >= 0
