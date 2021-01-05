# -*- coding: utf-8 -*-

from .split_model import SplitModel
from .gem import GEM
from .agem import AGEM

import lifelong.utils.resnet
import trojanvision.models

import torch.nn as nn
from collections import OrderedDict


class _ResNetcomp(trojanvision.models.resnet._ResNet):
    def __init__(self, layer: int = 18, **kwargs):
        super().__init__(layer=layer, **kwargs)
        _model = lifelong.utils.resnet.ResNet18(nclasses=self.num_classes)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', _model.conv1),
            ('bn1', _model.bn1),  # nn.BatchNorm2d(64)
            ('relu', nn.ReLU(inplace=True)),  # nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ('layer1', _model.layer1),
            ('layer2', _model.layer2),
            ('layer3', _model.layer3),
            ('layer4', _model.layer4)
        ]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', _model.linear)  # nn.Linear(512 * block.expansion, num_classes)
        ]))


class ResNetcomp(trojanvision.models.ResNetcomp, SplitModel):
    def __init__(self, name: str = 'resnetcomp', layer: int = 18,
                 model_class: type[_ResNetcomp] = _ResNetcomp, **kwargs):
        super().__init__(name=name, layer=layer, model_class=model_class, **kwargs)
