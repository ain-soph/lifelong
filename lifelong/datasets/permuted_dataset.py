#!/usr/bin/env python3

from .lifelong_dataset import LifelongDataset

from trojanzoo.environ import env
from trojanzoo.utils.data import TensorListDataset, dataset_to_list
from trojanvision.datasets import ImageSet

import torch
import torch.utils.data
import numpy as np


class PermutedDataset(LifelongDataset, ImageSet):
    def __init__(self, **kwargs):
        super().__init__(lifelong_type='permuted', **kwargs)
        num_classes = self.num_classes
        self.num_classes = self.task_num * num_classes
        self.class_order = torch.arange(self.num_classes).tolist()
        self.class_order_list: list[list[int]] = [a.tolist() for a in np.array_split(self.class_order, self.task_num)]

    def get_dataset_dict_fn(self) -> dict[str, list[torch.utils.data.Dataset]]:
        torch.manual_seed(env['seed'])  # TODO
        numel = self.data_shape[0] * self.data_shape[1] * self.data_shape[2]
        self.permuted_idx: list[torch.Tensor] = [torch.randperm(numel) for _ in range(self.task_num)]
        dataset = {
            'train': self.get_dataset(mode='train'),
            'valid': self.get_dataset(mode='valid'),
        }
        dataset_dict: dict[str, list[torch.utils.data.Dataset]] = {}
        for mode in ['train', 'valid']:
            data, target = dataset_to_list(dataset[mode])
            data = torch.stack(data)
            if data.dim() == 3:
                data = data.unsqueeze(1)
            flatten_data = data.flatten(1)
            dataset_dict[mode] = [TensorListDataset(flatten_data[:, idx].view_as(data),
                                                    (np.array(target) + task_id * self.num_classes).tolist()
                                                    ) for task_id, idx in enumerate(self.permuted_idx)]
        return dataset_dict
