#!/usr/bin/env python3

from .lifelong_model import LifelongModel
from .gem import GEM
from .agem import AGEM as MethodClass
from .ewc import EWC
from .icarl import ICARL

import trojanvision.models


class ResNetS(trojanvision.models.ResNetS, MethodClass):
    pass


class ResNet(trojanvision.models.ResNet, MethodClass):
    pass


class Net(trojanvision.models.Net, MethodClass):
    pass
