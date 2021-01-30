#!/usr/bin/env python3
from .lifelong_dataset import LifelongDataset

import torch
import torch.utils.data
import numpy as np


class SplitDataset(LifelongDataset):
    def __init__(self, class_order: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.param_list['split'] = ['class_order_list']
        if class_order is None:
            class_order = torch.arange(self.num_classes).tolist()
            # np.random.seed(env['seed'])
            # np.random.shuffle(class_order)
            # class_order = np.random.choice(class_order, len(class_order), replace=False)
        self.class_order_list: list[list[int]] = [a.tolist() for a in np.array_split(class_order, self.task_num)]

    def get_dataset_dict_fn(self) -> dict[str, list[torch.utils.data.Dataset]]:
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
        return dataset_dict
