#!/usr/bin/env python3

from .lifelong_model import LifelongModel
from .dgr_wgan_gp import Critic, Generator
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.environ import env
from trojanzoo.utils.influence import InfluenceFunction

import torch
from typing import TYPE_CHECKING
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.nn
    import torch.autograd
    import torch.optim
    import torch.utils.data


class DGR(LifelongModel):
    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('')
