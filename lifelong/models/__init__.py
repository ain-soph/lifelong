#!/usr/bin/env python3

from .lifelong_model import LifelongModel
from .concrete import *
from lifelong.configs import config, Config
import trojanvision.models

class_dict: dict[str, LifelongModel] = {
    'resnets': ResNetS,
    'resnet': ResNet,
    'net': Net
}


def add_argument(*args, config: Config = config, class_dict: dict[str, type[LifelongModel]] = class_dict, **kwargs):
    return trojanvision.models.add_argument(*args, config=config, class_dict=class_dict, **kwargs)


def create(*args, config: Config = config, class_dict: dict[str, type[LifelongModel]] = class_dict, **kwargs):
    return trojanvision.models.create(*args, config=config, class_dict=class_dict, **kwargs)
