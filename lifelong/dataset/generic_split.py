# coding: utf-8
from trojanzoo.datasets import Dataset


class GenericSplit(Dataset):
    def __init__(self, split_num: int, split_method: str = 'class', **kwargs):
        super().__init__(**kwargs)
        self.param_list['split'] = ['split_num', 'split_method']
        self.split_num = split_num
        self.split_method = split_method
        dataset_list = self.split_dataset(split_num, split_method)

    def split_dataset(self, mode: str) -> None:
        train_set = self.get_full_dataset()
        pass

    def get_split_loader_list(split_):
        pass
