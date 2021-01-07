#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from lifelong.datasets.split_dataset import SplitDataset
from trojanvision.models import ImageModel
from trojanzoo.utils.output import prints, ansi

import torch
import torch.utils.data
import torch.optim.lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from typing import Callable


class SplitModel(ImageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset: SplitDataset = self.dataset
        self.current_task: int = 0
        self.task_mask = torch.zeros(self.dataset.split_num, self.num_classes, dtype=torch.bool)
        for task_id in range(self.dataset.split_num):
            self.task_mask[task_id][torch.tensor(self.dataset.class_order_list[task_id])] = True
        label_to_task: dict[int, int] = {
            label: task for task, label_list in enumerate(self.dataset.class_order_list)
            for label in label_list}
        self.label_to_task = torch.tensor([label_to_task[label] for label in range(self.num_classes)])
        self.label_to_idx = torch.tensor(
            [sorted(self.dataset.class_order_list[self.label_to_task[label]]).index(label) for label in range(self.num_classes)])

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if _output is None:
            _output = self(_input, **kwargs)
        _task = self.label_to_task[_label]
        _mask = self.task_mask[_task]
        _output[~_mask] = -1e10
        _output = _output[_mask].view(len(_input), -1)
        _label = self.label_to_idx[_label].to(device=_label.device)
        return super().loss(_input=_input, _label=_label, _output=_output, **kwargs)

    def _train(self, *args, epoch: int = None, loader_train: list[torch.utils.data.DataLoader] = None,
               validate_interval: int = 10, save: bool = False, amp: bool = False,
               lr_scheduler=None, indent: int = 0, after_task_fn: Callable[..., None] = None, **kwargs):
        if after_task_fn is None and hasattr(self, 'after_task_fn'):
            after_task_fn = self.after_task_fn
        if loader_train is None:
            loader_train = self.dataset.loader['train']
        for task_id in range(self.dataset.split_num):
            self.current_task = task_id
            prints('{green}task{reset}: {0:d}'.format(task_id, **ansi), indent=indent)
            super()._train(*args, epoch=epoch, loader_train=loader_train[task_id],
                           validate_interval=validate_interval, save=save, amp=amp,
                           start_epoch=task_id * epoch, indent=indent + 10,
                           tag=str(task_id), lr_scheduler=lr_scheduler, **kwargs)
            if callable(after_task_fn):
                after_task_fn(task_id=task_id)

            # if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            #     lr_scheduler.step(0)

    def _validate(self, *args, loader=None, _epoch: int = None, tag: str = '', writer: SummaryWriter = None, **kwargs):
        loss = 0.0
        acc = 0.0
        accs = []
        if loader is None:
            loader = self.dataset.loader['valid']
        for tid in range(self.current_task + 1):
            loader = self.dataset.loader['valid'][tid]
            loss, acc = super()._validate(*args, loader=loader, tag=str(tid),
                                          print_prefix='Validate ' + str(tid),
                                          _epoch=_epoch, **kwargs)
            accs.append(acc)

        writer.add_scalar('Acc/', tag, torch.mean(accs).item(), _epoch)
        return loss, acc
