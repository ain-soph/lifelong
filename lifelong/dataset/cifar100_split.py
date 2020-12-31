# coding: utf-8
from .generic_split import GenericSplit
from trojanvision.datasets import CIFAR100
import torchvision.datasets as datasets
from typing import Union


class CIFAR100_Split(CIFAR100, GenericSplit):
    def __init__(self, split_num: int = 20, split_method: str = 'class', **kwargs):
        super().__init__(split_num=split_num, split_method=split_method, **kwargs)
