#!/usr/bin/env python3

from .permuted_dataset import PermutedDataset
from trojanvision.datasets import MNIST


class PermutedMNIST(MNIST, PermutedDataset):
    name = 'permuted_mnist'

    def __init__(self, task_num: int = 20, **kwargs):
        super().__init__(task_num=task_num, **kwargs)
