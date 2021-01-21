#!/usr/bin/env python3

from .split_model import SplitModel
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.environ import env
from trojanzoo.utils.influence import InfluenceFunction

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.nn
    import torch.autograd
    import torch.optim
    import torch.utils.data


class EWC(SplitModel):
    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--lambd', dest='lambda for each past task', type=float)
        group.add_argument('--sample_num', dest='sample batch numner', type=int)

    def __init__(self, *args, lambd:float = 10, sample_num:int =64, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['ewc'] = ['lambd', 'sample_num']
        self.lambd = lambd
        self.sample_num = sample_num
        self.influence = InfluenceFunction(model=self)
        self.memory_params:list[dict[str,torch.Tensor]]=[]
        self.memory_fims:list[dict[str,torch.Tensor]]=[]

    def after_task_fn(self, task_id: int):
        assert len(self.memory_fims)==task_id
        self.memory_params.append({name:param.flatten().detach().cpu() for name,param in self.named_parameters() if param.requires_grad})
        fims={}
        for i, data in enumerate(self.dataset.loader['train'][task_id]):
            if i<self.sample_num:
                _input, _label = self.get_data(data)
                result:torch.Tensor = F.log_softmax(self(_input)).max(dim=1)[0].mean()
                for name,param in self.named_parameters():
                    if param.requires_grad:
                        if name not in fims.keys():
                            fims[name]=[]
                        fim = torch.autograd.grad(result, param, retain_graph=True)**2
                        fims[name].append(fim)
        self.memory_fims.append({name: torch.stack(fim_list).mean(dim=0) for name,fim_list in fims.items()})
    
    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None, _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if _output is None:
            _output = self(_input, **kwargs)
        loss = self.criterion(_output, _label)
        current_param = {name:param.flatten() for name,param in self.named_parameters() if param.requires_grad}
        for task_id, (memory_param, memory_fim) in enumerate(zip(self.memory_params, self.memory_fims)):
            for layer in memory_param.keys():
                loss += self.lambd / 2 * memory_fim[layer] @ (current_param[layer]-memory_param[layer].to(current_param.device))**2
        return loss
