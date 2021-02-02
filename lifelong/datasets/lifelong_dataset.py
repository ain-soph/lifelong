#!/usr/bin/env python3
from trojanzoo.datasets import Dataset
from trojanzoo.environ import env

import torch
import torch.utils.data


class LifelongDataset(Dataset):
    def __init__(self, task_num: int, shuffle_tasks: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.param_list['lifelong'] = ['task_num', 'shuffle_tasks']
        self.task_num = task_num
        self.shuffle_tasks = shuffle_tasks
        self.task_idx: list[int] = torch.arange(self.task_num).tolist()
        if shuffle_tasks:
            self.param_list['lifelong'].append('task_idx')
            torch.manual_seed(env['seed'])  # TODO
            self.task_idx = torch.randperm(self.task_num).tolist()
        dataset_dict = self.get_dataset_dict()
        self.loader: dict[str, list[torch.utils.data.DataLoader]] = {
            'train': [self.get_dataloader(dataset=subset) for subset in dataset_dict['train']],
            'valid': [self.get_dataloader(dataset=subset) for subset in dataset_dict['valid']],
        }

    def get_dataset(self, *args, task_id: int = None, **kwargs) -> torch.utils.data.Dataset:
        dataset = super().get_dataset(*args, **kwargs)
        if task_id is None:
            return dataset
        return super().get_dataset(dataset=dataset, classes=self.class_order_list[task_id])

    def get_dataset_dict(self, **kwargs) -> dict[str, list[torch.utils.data.Dataset]]:
        dataset_dict = self.get_dataset_dict_fn(**kwargs)
        if self.shuffle_tasks:
            dataset_dict = {
                'train': [dataset_dict['train'][i] for i in self.task_idx],
                'valid': [dataset_dict['valid'][i] for i in self.task_idx],
            }
        return dataset_dict

    def get_dataset_dict_fn(self, **kwargs) -> dict[str, list[torch.utils.data.Dataset]]:
        pass
