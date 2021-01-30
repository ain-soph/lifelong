#!/usr/bin/env python3

from .split_model import SplitModel
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.environ import env
from trojanzoo.utils.influence import InfluenceFunction

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.nn
    import torch.autograd
    import torch.optim
    import torch.utils.data


class ICARL(SplitModel):
    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)

    def __init__(self, *args, lambd: float = 10, sample_num: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['ewc'] = ['lambd', 'sample_num']
        self.lambd = lambd
        self.sample_num = sample_num

    def update_representation(self):
        pass