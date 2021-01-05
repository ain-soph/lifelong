#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from trojanzoo.datasets import Dataset
from trojanzoo.environ import env

import torch.utils.data
import numpy as np


class SplitDataset(Dataset):
    def __init__(self, split_num: int, class_order: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['split'] = ['split_num', 'class_order_list']
        self.split_num = split_num
        if class_order is None:
            class_order = np.arange(self.num_classes)
            # np.random.seed(env['seed'])
            # np.random.shuffle(class_order)
            # class_order = np.random.choice(class_order, len(class_order), replace=False)
        self.class_order_list: list[list[int]] = [a.tolist() for a in np.array_split(class_order, split_num)]
        dataset = {
            'train': self.get_dataset(mode='train'),
            'valid': self.get_dataset(mode='valid'),
        }
        dataset_dict = {
            'train': [self.get_dataset(dataset=dataset['train'], classes=class_list)
                      for class_list in self.class_order_list],
            'valid': [self.get_dataset(dataset=dataset['valid'], classes=class_list)
                      for class_list in self.class_order_list],
        }
        self.loader: dict[str, list[torch.utils.data.DataLoader]] = {
            'train': [self.get_dataloader(dataset=subset) for subset in dataset_dict['train']],
            'valid': [self.get_dataloader(dataset=subset) for subset in dataset_dict['valid']],
        }

    def get_dataset(self, *args, task_id: int = None, **kwargs) -> torch.utils.data.Dataset:
        dataset = super().get_dataset(*args, **kwargs)
        if task_id is None:
            return dataset
        return super().get_dataset(dataset=dataset, classes=self.class_order_list[task_id])
