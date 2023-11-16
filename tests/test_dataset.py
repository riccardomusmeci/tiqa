import os

from path import DATASET_DIR

from tiqa.dataset import InferenceDataset, IQADataset
from tiqa.transform import DivTargetBy, Transform


def test_train_dataset() -> None:
    dataset = IQADataset(
        root_dir=DATASET_DIR,
        transform=Transform(train=True, input_size=224),
        target_transform=DivTargetBy(100),
    )

    img, target = dataset[0]
    assert img.shape == (3, 224, 224)
    assert target < 1


def test_inference_dataset() -> None:
    dataset = InferenceDataset(
        data_dir=os.path.join(DATASET_DIR, "images"),
        transform=Transform(train=False, input_size=224),
    )

    img, _ = dataset[0]
    assert img.shape == (3, 224, 224)
    assert len(dataset) == 3
