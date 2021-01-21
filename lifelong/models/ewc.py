#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .split_model import SplitModel
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.environ import env
from trojanzoo.utils.influence import InfluenceFunction
from collections import OrderedDict

import torch
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
        self.optim_params = OrderedDict()
        self.influence = InfluenceFunction(model=self)

    def update_optim_param(self): 
        k = "task_" + str(len(self.optim_params))
        self.optim_params.update({k, [param for param in self.model.parameters()]})
        
    def ewc_reg(self):
        """
            Calculate the EWC regularizer, 
            append it with the original loss.
        """
        params = [param for param in self.model.parameters()]           
        FIMs = [self.influence.compute_fim(param) for param in self.model.parameters()]  # todo: calculated by current params or the optim
        
        ewc_reg = 0 
        for k, optim_params in self.optim_params.items():  # for each past task
            for i, fim in enumerate(FIMs):
                ewc_reg += (self.lam/2) * torch.sum(torch.mul(fim[i], 
                        torch.square(params[i] - optim_params[i])))
        
        return ewc_reg

    def _train(self):
        # step1: training with loss+ewc_reg()

        # update optim params after training
        self.update_optim_param(self)

    