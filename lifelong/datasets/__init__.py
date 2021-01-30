#!/usr/bin/env python3

from .lifelong_dataset import LifelongDataset
from .split_dataset import SplitDataset
from .permuted_dataset import PermutedDataset

from .permuted_mnist import PermutedMNIST
from .split_cifar100 import SplitCIFAR100

import trojanvision.datasets

class_dict: dict[str, LifelongDataset] = {
    'permuted_mnist': PermutedMNIST,
    'split_cifar100': SplitCIFAR100,
}


def add_argument(*args, class_dict: dict[str, type[LifelongDataset]] = class_dict, **kwargs):
    return trojanvision.datasets.add_argument(*args, class_dict=class_dict, **kwargs)


def create(*args, class_dict: dict[str, type[LifelongDataset]] = class_dict, **kwargs):
    return trojanvision.datasets.create(*args, class_dict=class_dict, **kwargs)
