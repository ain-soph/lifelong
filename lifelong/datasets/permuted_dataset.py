#!/usr/bin/env python3

from .lifelong_dataset import LifelongDataset

from trojanzoo.environ import env
from trojanzoo.utils.data import TensorListDataset, dataset_to_list
from trojanvision.datasets import ImageSet

import torch
import torch.utils.data


class PermutedDataset(LifelongDataset, ImageSet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        torch.manual_seed(env['seed'])  # TODO
        numel = self.data_shape[0] * self.data_shape[1] * self.data_shape[2]
        self.permuted_idx: list[torch.Tensor] = [torch.randperm(numel)]

    def get_dataset_dict_fn(self) -> dict[str, list[torch.utils.data.Dataset]]:
        dataset = {
            'train': self.get_dataset(mode='train'),
            'valid': self.get_dataset(mode='valid'),
        }
        dataset_dict: dict[str, list[torch.utils.data.Dataset]] = {}
        for mode in ['train', 'valid']:
            data, target = dataset_to_list(dataset[mode])
            data = torch.stack(data)
            flatten_data = data.flatten(1)
            dataset_dict[mode] = [TensorListDataset(flatten_data[:, idx].view_as(data),
                                                    target) for idx in self.permuted_idx]
        return dataset_dict
