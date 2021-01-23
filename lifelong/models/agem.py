#!/usr/bin/env python3

from .gem import GEM
from trojanvision.optim import PGD
from trojanzoo.environ import env
from trojanzoo.utils.data import sample_batch, dataset_to_list

import torch
import torch.nn
import torch.optim
import torch.utils.data
import argparse


class AGEM(GEM):

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--sample_size', dest='sample_size', type=int)

    def __init__(self, *args, sample_size: int = 1300, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['agem'] = ['sample_size']
        self.sample_size = sample_size

    # def sample_batch(self, dataset: torch.utils.data.Dataset, batch_size: int) -> tuple[list, list[int]]:
    #     _, targets = dataset_to_list(dataset, label_only=True)
    #     class_list = list(set(targets))
    #     targets = torch.tensor(targets)
    #     perm = torch.randperm(len(targets))
    #     idx = torch.cat([perm[(targets[perm] == _class).nonzero(as_tuple=True)[0][:batch_size]]
    #                      for _class in class_list])
    #     return sample_batch(dataset, idx=idx)

    def store_grad(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_grad = torch.cat([param.grad.flatten() for param in self.params])
        self.zero_grad()

        memory_data = torch.cat(self.memory_data[:self.current_task])
        memory_targets = torch.cat(self.memory_targets[:self.current_task])
        idx = torch.randperm(len(memory_targets))[:self.sample_size]
        memory_data = memory_data[idx].to(device=env['device'])
        memory_targets = memory_targets[idx].to(device=env['device'])
        if self.pgd is not None:
            pgd: PGD = self.pgd

            def loss_fn_new(X: torch.FloatTensor) -> torch.Tensor:  # TODO: use functools.partial
                return -self.loss(X, memory_targets)    # TODO: loss_fn
            memory_adv_data, _ = pgd.optimize(_input=memory_data, loss_fn=loss_fn_new)
            memory_data = memory_adv_data
            # memory_data = torch.cat([memory_data, memory_adv_data])
            # memory_targets = memory_targets.repeat(2)
        memory_loss = self.loss(_input=memory_data, _label=memory_targets)
        memory_loss.backward()
        prev_grad = torch.cat([param.grad.flatten() for param in self.params])
        self.zero_grad()
        return current_grad, prev_grad

    @staticmethod
    def project2cone2(current_grad: torch.Tensor, prev_grad: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dot_result = current_grad @ prev_grad
            if dot_result > 0:
                return current_grad
            grad = current_grad - (dot_result / prev_grad.pow(2).sum()) * prev_grad
            return grad
            # print(grad.dot(current_grad) / grad.norm(p=2) / current_grad.norm(p=2))
            # print(grad.dot(prev_grad) / grad.norm(p=2) / prev_grad.norm(p=2))
            # print(prev_grad.dot(current_grad) / prev_grad.norm(p=2) / current_grad.norm(p=2))
            # print()
