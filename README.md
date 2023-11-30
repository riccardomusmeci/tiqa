<p align="center">
  <img width="200" height="200" src="static/tiqa.png">
</p>

----

**tiqa** is a PyTorch image quality assessment library with support to PyTorch-Lightning and easy access to experiment with your own dataset.

## **How to install 🔨**
```
git clone https://github.com/riccardomusmeci/tiqa
cd tiqa
pip install .
```

## **Models 🤖**
tiqa actually supports two models:
- [Deep Blind Image Quality Assessment Using a Deep Neural Network](https://arxiv.org/abs/1809.00219)
- [Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild](https://arxiv.org/abs/2304.00451)

To use a model, just call `create_model` function with the model name and the pretrained weights path (if you want to use a pretrained version).

```python
from tiqa.model import create_model

model = create_model(
    model_name="dbcnn_vgg16",
    ckpt_path="PATH/TO/CKPT",
)
```

### **Pretrained Weight 🏋️**


| Model | Dataset | SRCC | PLCC | Weights | Notes |
|-------|--------------|------|------|------| ------|
| dbcnn_vgg16 | Koniq10k | - | - | [gdrive](https://drive.google.com/file/d/13GWi9ka1z7ywo04_TGGLYzY6Id_NaBFO/view?usp=share_link) |
| dbcnn_vgg16 | - | - | - | [gdrive](https://drive.google.com/file/d/1rqeopYS38XqiWcZ8xoa1PxwqEhM4SMgU/view?usp=share_link) |
| reiqa_resnet50 | Koniq10k | - | - | [gdrive](https://drive.google.com/file/d/1PH39CLb7s2RDAripFoTfjanTljmvQDJe/view?usp=share_link) | Trained on Koniq10k from [initial quality aware weights](https://drive.google.com/file/d/1DYMx8omn69yXUmBFL728JD3qMLNogFt8/view?usp=sharing) of [repository](https://github.com/avinabsaha/ReIQA/tree/main)|


## **Concepts 💡**
tiqa tries to avoid writing again, again, and again (and again) the same code to train, test and make predictions with a image classification model.

tiqa works in three different ways:
* fully automated with configuration files 🚀
* semi-automated with full support to PyTorch Lightning ⚡️
* I-want-to-write-my-own-code-but-also-using-tiqa 🧑‍💻

### **TiqaConfiguration 📄**
With TiqaConfiguration file you don't need to write any code for training an inference.

A configuration file is like the on in config/config.yaml.

## **Train**

### **Dataset Structure**
tiqa dataset must have the following structure:
```
dataset
      |__train
      |       |__images
      |       |        |__img_1.jpg
      |       |        |__img_2.jpg
      |       |        |__ ...
      |       |__annotations.csv
      |____val
              |__images
              |        |__img_1.jpg
              |        |__img_2.jpg
              |        |__ ...
              |__annotations.csv
```


### **Fully Automated 🚀**
Once configuration experiment file is ready, just use tiqa like this:

```python
from tiqa.core import train

train(
    config_path="PATH/TO/CONFIG.YAML",
    train_data_dir="PATH/TO/TRAIN/DATA/DIR",
    val_data_dir="PATH/TO/VAL/DATA/DIR",
    output_dir="PATH/TO/OUTPUT/DIR",
    resume_from="PATH/TO/CKPT/TO/RESUME/FROM", # this is when you want to start retraining from a Lightning ckpt
)
```

### **Semi-Automated ⚡️**
tiqa delivers some pre-built modules based on PyTorch-Lightning to speed up experiments.

```python
from tiqa.model import create_model
from tiqa.transform import Transform
from tiqa.loss import create_criterion
from tiqa.optimizer import create_optimizer
from tiqa.lr_scheduler import create_lr_scheduler
from tiqa.pl import create_callbacks
from pytorch_lightning import Trainer
from tiqa.pl import from ..pl import IQADataModule, IQAModelModule

# Setting up datamodule, model, callbacks, logger, and trainer
datamodule = IQADataModule(
    train_data_dir=...,
    val_data_dir=...,
    train_transform=Transform(train=True, ...),
    val_transform=Transform(train=False, ...),
    engine="pil", # or "cv2"
    batch_size=16,
    ...
)
model = create_model(
    "dbcnn_vgg16",
    ckpt_path=... # load a pretrained-version
)
criterion = create_criterion("mse")
optimizer = create_optimizer(params=model.parameters(), optimizer="sgd", lr=.001, ...)
lr_scheduler = create_lr_scheduler(optimizer=optimizer, ...)
pl_model = IQAModelModule(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    unfreeze_after=10 # unfreeze model after 10 epochs
)
callbacks = create_callbacks(output_dir=..., ...)
trainer = Trainer(callbacks=callbacks, ...)

# Training
trainer.fit(model=pl_model, datamodule=datamodule)
```

### **I want to write my own code 🧑‍💻**
Use tiqa `ImageFolderDataset`, `Transform`, and `create_stuff` functions to write your own training loop.

```python
from tiqa.transform import Transform
from tiqa.dataset import IQADataset
from tiqa.model import create_model
from tiqa.loss import create_criterion
from tiqa.optimizer import create_optimizer
from torch.utils.data import DataLoader
import torch

train_dataset = IQADataset(
    root_dir=...,
    transform=Transform(train=True, input_size=224),
    engine="pil", # or "cv2"
)
train_dl = DataLoader(dataset=train_dataset, batch_size=16)

model = create_model(
    model_name="dbcnn_vgg16",
    ckpt_path=..., # load a pretrained-version
    freeze_encoder=False,
    freeze_scnn=True
)
criterion = create_criterion(name="mse")
optimizer = create_optimizer(params=model.parameters(), name="sgd", lr=0.0005)

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in train_dl:
        optimizer.zero_grad()
        x, target = batch
        logits = model(x)
        loss = criterion(logits, target.view(logits.shape, 1))
        loss.backward()
        optimizer.step()
```
<!--
## **Inference 🧐**
Also in inference mode, you can pick between "fully automated", "semi-automated", "write my own code" mode.


### **Fully Automated 🚀**
Once the train is over, you'll find a *config.yaml* file merging all the setups from different sections.

```python
from tiqa.core import predict

predict(
    ckpt_path="PATH/TO/OUTPUT/DIR/checkpoints/model.ckpt",
    config_path="PATH/TO/OUTPUT/DIR/config.yaml",
    images_dir="PATH/TO/IMAGES",
    output_dir="PATH/TO/OUTPUT/DIR/predictions", # you can choose your own path
    apply_gradcam=True, # save gradcam images
    gradcam_with_preds=True, # if True, split gradcam images based on model predicitons
    layer="...", # layer to use for gradcam
)
``` -->
