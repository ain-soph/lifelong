#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .split_model import SplitModel
from .gem import GEM
from .agem import AGEM

from trojanvision.models.resnet import _ResNetS, ResNetS


class ResNetS(ResNetS, AGEM):
    def __init__(self, name: str = 'resnets', layer: int = 18,
                 model_class: type[_ResNetS] = _ResNetS, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)
