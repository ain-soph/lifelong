#!/usr/bin/env python3

from .split_dataset import SplitDataset
from trojanvision.datasets import CIFAR100


class SplitCIFAR100(CIFAR100, SplitDataset):
    name = 'split_cifar100'

    def __init__(self, task_num: int = 20, **kwargs):
        super().__init__(task_num=task_num, **kwargs)

    # @staticmethod
    # def get_transform(mode: str) -> transforms.ToTensor:
    #     return transforms.ToTensor()
