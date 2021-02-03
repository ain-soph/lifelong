#!/usr/bin/env python3

from .split_model import SplitModel
from .concrete import ResNetS, ResNet
import trojanvision.models

class_dict: dict[str, SplitModel] = {
    'resnets': ResNetS,
    'resnet': ResNet,
}


def add_argument(*args, class_dict: dict[str, type[SplitModel]] = class_dict, **kwargs):
    return trojanvision.models.add_argument(*args, class_dict=class_dict, **kwargs)


def create(*args, class_dict: dict[str, type[SplitModel]] = class_dict, **kwargs):
    return trojanvision.models.create(*args, class_dict=class_dict, **kwargs)
