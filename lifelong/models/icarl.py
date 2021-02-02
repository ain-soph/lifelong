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
        # [class_id], (split_mem_size, C, H, W)
        for i in range(task_id):    # TODO
            for j in self.dataset.class_order_list[i]:
                self.memory_data[j] = self.memory_data[j][:split_mem_size]
                self.memory_targets[j] = self.memory_targets[j][:split_mem_size]

        for i in range(len(self.memory_data)):
            self.memory_data[i] = self.memory_data[i][:split_mem_size]
            self.memory_targets[i] = self.memory_targets[i][:split_mem_size]
        
        # add samples for current task

        for y in labelset:
            inputs = self.get_data_by_label(self.dataset.loader['train'][task_id], y)
            y_feats: torch.Tensor = self.get_final_fm(inputs)  # (N, D)
            mean_fm: torch.Tensor = torch.mean(y_feats, dim=0) # (1, D)

            found_ids = []
            rest_ids = list(range(y_feats.shape[0]))
            id_list = torch.arange(len(rest_ids))
            found_fm_sum: torch.Tensor = torch.zeros(y_feats.shape[1], device=y_feats.device)
            if len(rest_ids) <= split_mem_size:
                found_ids = rest_ids
            else:
                while len(found_ids < split_mem_size):
                    fms = (y_feats[rest_ids] + found_fm_sum) / (len(found_ids) + 1)  # (N-found, D)
                    _id = int(torch.cdist(fms, mean_fm, p=2).argmin())
                    org_id = int(id_list[_id])
                    found_ids.append(org_id)
                    rest_ids = list(set(rest_ids) - {org_id})
                    found_fm_sum += y_feats[org_id]
                    id_list[org_id:] += 1
            self.memory_data[self.label_to_task[y]].append(y_input[found_ids])  # TODO
            self.memory_targets.append([found_ids] * y)

        # Calculate q
        for data in self.dataset.loader['train'][self.current_task + 1]:
            _input, _label = self.get_data(data)
            idx = self.indices
            self.q[idx] = F.sigmoid(self(_input)).to(self.q.device)

        # update loader # TODO
        if self.current_task < self.dataset.task_num - 1:
            org_dataset = self.dataset.loader['train'][self.current_task + 1].dataset
            self.memory_data
