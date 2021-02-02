#!/usr/bin/env python3

from .split_dataset import SplitDataset
from trojanvision.datasets import CUB200, CUB200_2011


class SplitCUB200(CUB200, SplitDataset):
    name = 'split_cub200'

    def __init__(self, task_num: int = 20, **kwargs):
        super().__init__(task_num=task_num, **kwargs)


class SplitCUB200_2011(CUB200_2011, SplitDataset):
    name = 'split_cub200_2011'

    def __init__(self, task_num: int = 20, **kwargs):
        super().__init__(task_num=task_num, **kwargs)
