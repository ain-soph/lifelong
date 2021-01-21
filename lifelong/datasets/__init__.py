#!/usr/bin/env python3

from .split_dataset import SplitDataset
from .cifar100 import CIFAR100
import trojanvision.datasets

class_dict: dict[str, SplitDataset] = {
    'cifar100': CIFAR100,
}


def add_argument(*args, class_dict: dict[str, type[SplitDataset]] = class_dict, **kwargs):
    return trojanvision.datasets.add_argument(*args, class_dict=class_dict, **kwargs)


def create(*args, class_dict: dict[str, type[SplitDataset]] = class_dict, **kwargs):
    return trojanvision.datasets.create(*args, class_dict=class_dict, **kwargs)
