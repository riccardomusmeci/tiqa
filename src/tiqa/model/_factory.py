import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .factory import daclip_deg_vit_base_patch32_224, dbcnn_vgg16

_FACTORY = {"dbcnn_vgg16": dbcnn_vgg16, "daclip_deg_vit_base_patch32_224": daclip_deg_vit_base_patch32_224}


def create_model(  # type: ignore
    model_name: str,
    ckpt_path: Optional[Union[Path, str]] = None,
    prefix_key: Optional[str] = None,
    to_replace: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    """Create an IQA model.

    Args:
        model_name (str): model name.
        ckpt_path (Optional[Union[Path, str]], optional): path to pretrained checkpoint. Defaults to None.
        prefix_key (Optional[str], optional): prefix key for loading state dict. Defaults to None.
        to_replace (Optional[str], optional): string to replace in state dict keys. Defaults to None.

    Returns:
        nn.Module: model
    """

    if model_name in list(_FACTORY.keys()):
        model = _FACTORY[model_name](**kwargs)  # type: ignore
    else:
        print(f"> [ERROR] No model named {model_name} found in factory")
        quit()

    if ckpt_path is not None:
        state_dict = load_ckpt(ckpt_path=ckpt_path, prefix_key=prefix_key, to_replace=to_replace)
        model.load_state_dict(state_dict=state_dict)
        print(f"> Loaded state dict from {ckpt_path}")

    return model


def load_ckpt(ckpt_path: Union[Path, str], prefix_key: Optional[str] = None, to_replace: Optional[str] = None) -> Dict:
    """Load checkpoint.

    Args:
        ckpt_path (Union[Path, str]): path to checkpoint.
        prefix_key (Optional[str], optional): prefix key for loading state dict. Defaults to None.
        to_replace (Optional[str], optional): string to replace in state dict keys. Defaults to None.

    Returns:
        Dict: state dict
    """

    assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist"
    try:
        state_dict = torch.load(ckpt_path)
    except RuntimeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if prefix_key is not None:
        state_dict = {prefix_key + k: w for k, w in state_dict.items()}

    if to_replace is not None:
        state_dict = {k.replace(to_replace, ""): w for k, w in state_dict.items()}

    return state_dict
