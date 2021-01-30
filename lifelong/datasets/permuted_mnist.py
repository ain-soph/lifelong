#!/usr/bin/env python3

from .permuted_dataset import PermutedDataset
from trojanvision.datasets import MNIST


class PermutedMNIST(MNIST, PermutedDataset):
    name = 'permuted_mnist'

    def __init__(self, task_num: int = 23, flag: bool = False, class_order: list[int] = None, **kwargs):
        if flag and class_order is None:
            class_order = []
        super().__init__(task_num=task_num, class_order=class_order, **kwargs)
