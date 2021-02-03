#!/usr/bin/env python3

from .split_model import SplitModel as MethodClass
from .gem import GEM
from .agem import AGEM
from .ewc import EWC
from .icarl import ICARL

from trojanvision.models.resnet import ResNetS, ResNet


class ResNetS(ResNetS, MethodClass):
    pass


class ResNet(ResNet, MethodClass):
    pass