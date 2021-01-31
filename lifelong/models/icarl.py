#!/usr/bin/env python3

from .split_model import SplitModel
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.environ import env
from trojanzoo.utils.influence import InfluenceFunction

import torch
import torch.nn as nn
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
        group.add_argument('--memory_size', dest='memory_size', type=int)

    def __init__(self, *args,  memory_size: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['icarl'] = []
        self.memory_size = memory_size
        self.memory_data: list[list[torch.Tensor]] = []
        self.memory_targets: list[list[torch.Tensor]] = []
        self.criterion = nn.BCEWithLogitsLoss() 
        self.past_labels: int = 0

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None, _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        loss = super().loss(_input, _label, _output=_output, **kwargs)
        for mem_imgs, mem_tars in zip(self.memory_data, self.memory_targets):
            for x, y in zip(mem_imgs, mem_tars):
                loss += self.criterion(x, y)
        return loss

    def after_task_fn(self, task_id: int):
        # count label numbers for current task
        labelset = set()
        for data in self.dataset.loader['train'][task_id]:
            _input, _label = self.get_data(data)
            labelset =  labelset | set(_label.tolist())
        self.past_labels += len(labelset)
        split_mem_size = self.memory_size / self.past_labels
        
        # reduce sample number for past tasks

        for i in range(len(self.memory_data)):
            self.memory_data[i] = self.memory_data[i][:split_mem_size]
            self.memory_targets[i] = self.memory_targets[i][:split_mem_size]
        
        # add samples for current task

        for y in labelset:
            inputs = self.get_data_by_label(self.dataset.loader['train'][task_id], y)
            featmaps: torch.Tensor = self.get_final_fm(inputs)  # (N, D)
            mean_fm: torch.Tensor = torch.mean(featmaps, dim=0) # (1, D)

            found_ids = []
            rest_ids = list(range(featmaps.shape[0]))
            found_fm: torch.Tensor = 0.0
            while len(found_ids<split_mem_size):
                _ids = self.dist_calc(mean_fm, featmaps[rest_ids], found_fm, len(found_ids)+1)
                found_ids.append(_ids)
                rest_ids = list(set(rest_ids) - set(found_ids))
                found_fm += featmaps[_ids]
            self.memory_data.append(featmaps[found_ids])
            self.memory_targets.append(torch.stack([y]*len(found_ids)))

    def get_data_by_label(self, loader: torch.utils.data.DataLoader, target_label: int):
        dataset = loader.dataset
        _input, _label = dataset_to_list(dataset)
        return torch.stack(_input)[torch.tensor(_label)==target_label]

    def dist_calc(self, mean_fn: torch.Tensor, fms: torch.Tensor, found_fm: torch.Tensor, k: int):
        fms = (fms + found_fm)/k  # (N', D) + (1, D) = (N', D)
        return torch.argmin(torch.cdist(fms, mean_fn, p=2)).item()