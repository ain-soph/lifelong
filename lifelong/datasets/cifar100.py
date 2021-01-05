# coding: utf-8

from .split_dataset import SplitDataset
from trojanvision.datasets import CIFAR100

import torchvision.transforms as transforms


class CIFAR100(CIFAR100, SplitDataset):
    name = 'cifar100'

    def __init__(self, split_num: int = 20, flag: bool = False, class_order: list[int] = None, **kwargs):
        if flag and class_order is None:
            class_order = []
        super().__init__(split_num=split_num, class_order=class_order, **kwargs)

    @staticmethod
    def get_transform(mode: str) -> transforms.ToTensor:
        return transforms.ToTensor()
