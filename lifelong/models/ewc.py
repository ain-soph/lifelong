#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .split_model import SplitModel
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.utils.influence import InfluenceFunction
from trojanzoo.environ import env

import torch
from kmeans_pytorch import kmeans
import numpy as np
import copy

from typing import TYPE_CHECKING
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.nn
    import torch.autograd
    import torch.optim
    import torch.utils.data


class EWC(SplitModel):
    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--lam', dest='lambda for each past task', type=float)

    def __init__(self, *args, lam: float = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['ewc'] = ['lam']
        self.lam = lam
        self.param_list = []
        self.FIMs = {}

    def update_param_list(self):
        self.param_list = [param for param in None]  # todo: get all params in model

    def star(self):
        self.star_params = []
        for v in range(len(self.param_list)):
            self.star_params.append(self.param_list[v].clone())
            
        name = 'task' + str(len(self.optim_params)+1)
        self.optim_params.update({name: copy.deepcopy(self.param_list)})
        
    def ewc_reg(self, lam=10):
        """Calculate the EWC regularizer, 
           append it with original loss
        """
        ewc_reg = 0 
        for k in self.FIMs:
            F_accum = self.FIMs[k]
            star_params = self.optim_params[k]
            for v in range(len(self.param_list)):
                ewc_reg += (lam/2) * torch.sum(torch.mul(F_accum[v], 
                 torch.square(self.param_list[v] - star_params[v])))
        
        return ewc_reg

    def _train(self):
        # step1: get all optim params
        # step2: update_param_list(self)
        # step3: update self.FIMs
        # step4: train_loss += ewc_reg(self.lam)
        pass

    