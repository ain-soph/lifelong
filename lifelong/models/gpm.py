#!/usr/bin/env python3

from .lifelong_model import LifelongModel
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.environ import env

import torch

from typing import TYPE_CHECKING
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.nn
    import torch.autograd
    import torch.optim
    import torch.utils.data


class GPM(LifelongModel):

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--memory_size', dest='memory_size', type=int)
        group.add_argument('--eps_base', dest='eps_base', type=float)
        group.add_argument('--eps_increase', dest='eps_increase', type=float)

    def __init__(self, *args, memory_size: int = 125, eps_base: float = 0.97, eps_increase: float = 0.003, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['gpm'] = ['memory_size', 'eps_th', 'eps_increase']
        self.memory_size = memory_size
        self.eps_base = eps_base
        self.eps_increase = eps_increase
        self.memory_data: dict[str, torch.Tensor] = {}    # (N, D)
        self.memory_square: dict[str, torch.Tensor] = {}    # (D, D)
        self.name_list: list[str] = [name.rsplit('.', maxsplit=1)[0]
                                     for name, _ in self.named_parameters() if 'weight' in name]

    def after_task_fn(self, task_id: int):
        eps_th = self.eps_base + task_id * self.eps_increase
        dataset = self.dataset.loader['train'][task_id].dataset
        data, _ = self.sample_batch(dataset, batch_size=self.memory_size)
        with torch.no_grad():
            layer_dict = self.get_all_layer(torch.stack(data).to(env['device']))    # TODO
        layer_dict = {key: value for key, value in layer_dict.items() if key in self.name_list}
        for layer in layer_dict.keys():
            if layer in self.memory_square.keys():
                result = self.memory_square[layer] @ layer_dict[layer]
                layer_dict[layer] -= result.view_as(layer_dict[layer])
                u, s, v = layer_dict[layer].svd()
                threshold = eps_th * layer_dict[layer].norm(P='fro')  # Frobenius norm
                for k in range(len(s)):
                    result = (u[:, :k] @ torch.diag_embed(s[:k]) @ v.t()[:k]).norm(P='fro')
                    if result > threshold:
                        self.memory_data[layer] = torch.cat([self.memory_data[layer], u[:, :k]])  # TODO
                        break
        self.memory_square = {key: value.t().dot(value) for key, value in self.memory_data.items()}  # (D, D)

    def sample_batch(self, dataset: torch.utils.data.Dataset, batch_size: int) -> tuple[list, list[int]]:
        _, targets = dataset_to_list(dataset, label_only=True)
        class_list = list(set(targets))
        _input = []
        _label: list[int] = []
        for _class in class_list:
            subset = self.dataset.get_class_set(dataset, [_class])
            current_input, current_label = sample_batch(subset, batch_size=batch_size)
            _input.extend(current_input)
            _label.extend(current_label)
        return _input, _label

    def after_loss_fn(self, optimizer: torch.optim.Optimizer,
                      _iter: int = None, total_iter: int = None, **kwargs):
        if self.current_task > 0:
            current_grads: dict[str, torch.Tensor] = {name.rsplit('.', maxsplit=1)[0]: param.grad
                                                      for name, param in self.named_parameters()
                                                      if param.grad is not None and 'weight' in name}
            # Project
            with torch.no_grad():
                for name in current_grads.keys():
                    result = self.memory_square[name] @ current_grads[name].flatten(1)
                    current_grads[name] -= result.view_as(current_grads[name])
            # Rewrite
            for name, param in self.named_parameters():
                if name in current_grads.keys():
                    param.grad.data.copy_(current_grads[name].view_as(param))
