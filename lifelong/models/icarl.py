#!/usr/bin/env python3

from .split_model import SplitModel
from trojanzoo.utils.data import IndexDataset, dataset_to_list
from trojanzoo.utils.tensor import onehot_label
from trojanzoo.environ import env

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

    def __init__(self, *args, memory_size: int = 2000, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['icarl'] = []
        self.memory_size = memory_size
        self.memory_data: list[torch.Tensor] = [torch.empty(self.memory_size, self.dataset.data_shape)]  # TODO
        self.memory_targets: list[torch.Tensor] = []  # TODO
        self.past_labels: int = 0

        for mode, loader_list in self.dataset.loader.items():
            for i, loader in enumerate(loader_list):
                data, targets = dataset_to_list(loader.dataset)
                dataset = IndexDataset(data, targets)
                self.dataset.loader[mode][i] = self.dataset.get_dataloader(mode=mode, dataset=dataset)
        self.indices: torch.Tensor = []    # temp variable
        self.q: torch.Tensor = torch.zeros(len(self.dataset.get_full_dataset('train')), self.num_classes)

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        self.indices = data[0]
        return super().get_data(data[1:], **kwargs)

    def define_criterion(self, **kwargs) -> nn.BCEWithLogitsLoss:
        if 'weight' not in kwargs.keys():
            kwargs['weight'] = self.loss_weights
        return nn.BCEWithLogitsLoss(**kwargs)

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if _output is None:
            _output = self(_input, **kwargs)
        _task = torch.full([_label.shape[0]], self.current_task)
        current_output, task_label = self.prune_output_and_label(_output, _label, _task=_task)
        task_num_classes = _output.shape[1]
        _onehot_label = onehot_label(task_label, task_num_classes)
        _onehot_label[~np.isin(_label, self.dataset.class_order_list[self.current_task])] = 0
        loss = self.criterion(_onehot_label, current_output)

        _mask = torch.zeros(self.num_classes).bool()
        for task_id in range(self.current_task):
            _mask |= self.task_mask[task_id]  # (num_classes)
        _mask = _mask.repeat(_label.shape[0], 1)
        prev_output, task_label = self.prune_output_and_label(_output, _label, _mask=_mask)
        _q: torch.Tensor = self.q[self.indices]    # (N, num_classes)
        _q = _q[_mask].view(_label.shape[0], -1).to(prev_output.device)
        loss += self.criterion(_q, prev_output)

        return loss

    def after_task_fn(self, task_id: int):
        # count label numbers for current task
        data, targets = dataset_to_list(self.dataset.loader['train'][task_id].dataset)
        labelset = set(targets)
        self.past_labels += len(labelset)
        split_mem_size = self.memory_size / self.past_labels

        # reduce sample number for past tasks
        # [class_id], (split_mem_size, C, H, W)
        for i in range(task_id):    # TODO
            for j in self.dataset.class_order_list[i]:
                self.memory_data[j] = self.memory_data[j][:split_mem_size]
                self.memory_targets[j] = self.memory_targets[j][:split_mem_size]

        # add samples for current task
        for y in labelset:
            y_input = torch.stack(data)[torch.tensor(targets) == y]
            y_feats = self.get_final_fm(y_input.to(env['device']))  # (N, D)    # TODO: iterative loader
            y_feats: torch.Tensor = y_feats / y_feats.norm(dim=1, keepdim=True)  # Normalize
            mean_fm = y_feats.mean(dim=0, keepdim=True)  # (1, D)
            mean_fm: torch.Tensor = mean_fm / mean_fm.norm(dim=1, keepdim=True)  # Normalize

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
