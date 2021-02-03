#!/usr/bin/env python3

from trojanvision.models import ImageModel
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils.output import prints, ansi

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from typing import TYPE_CHECKING
from typing import Callable
from lifelong.datasets.split_dataset import SplitDataset    # TODO: python 3.10
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
if TYPE_CHECKING:
    import torch.utils.data

# CUDA_VISIBLE_DEVICES=0 python train.py --verbose 1 --color --tqdm --flush_secs 20 --dataset split_cifar100 --model resnets --weight_decay 0.0 --momentum 0.0 --validate_interval 1 --batch_size 10 --epoch 8 --lr 0.03 --log_dir /data/rbp5354/log/robust_agem_cluster --tensorboard --adv_train


class SplitModel(ImageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset: SplitDataset
        self.current_task: int = 0
        self.task_mask = torch.zeros(self.dataset.task_num, self.num_classes, dtype=torch.bool)
        for task_id in range(self.dataset.task_num):
            self.task_mask[task_id][torch.tensor(self.dataset.class_order_list[task_id])] = True
        label_to_task: dict[int, int] = {
            label: task for task, label_list in enumerate(self.dataset.class_order_list)
            for label in label_list}
        self.label_to_task = torch.tensor([label_to_task[label] for label in range(self.num_classes)])
        self.label_to_idx = torch.tensor(
            [sorted(self.dataset.class_order_list[self.label_to_task[label]]).index(label) for label in range(self.num_classes)])
        self.params: list[list[nn.Parameter]] = []
        self.param_numels: list[int] = []

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if _output is None:
            _output = self(_input, **kwargs)
        _output, _label = self.prune_output_and_label(_output, _label)
        return super().loss(_input=_input, _label=_label, _output=_output, **kwargs)

    # TODO
    def prune_output_and_label(self, _output: torch.Tensor, _label: torch.Tensor,
                               _task: torch.Tensor = None, _mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if _mask is None:
            _task = _task if _task is not None else self.label_to_task[_label]  # (N)
            _mask = self.task_mask[_task]   # (N, num_classes)
        _output = _output[_mask].view(len(_label), -1)  # (N, task_num_classes) flatten and then view back
        _label = self.label_to_idx[_label].to(device=_label.device)  # (N) 0--task_num_classes
        return _output, _label

    def accuracy(self, _output: torch.Tensor, _label: torch.Tensor,
                 topk: tuple[int] = (1, 5)) -> tuple[float, ...]:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            _task = self.label_to_task[_label]  # (N)
            _mask = self.task_mask[_task]
            _output = _output.clone()
            _output[~_mask] = -1e10
            maxk = min(max(topk), self.num_classes)
            batch_size = _label.size(0)
            _, pred = _output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(_label[None])
            res: list[float] = []
            for k in topk:
                if k > self.num_classes:
                    res.append(100.0)
                else:
                    correct_k = float(correct[:k].sum(dtype=torch.float32))
                    res.append(correct_k * (100.0 / batch_size))
            return res

    def _train(self, epoch: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
               loader_train: list[torch.utils.data.DataLoader] = None, loader_valid: list[torch.utils.data.DataLoader] = None,
               after_task_fn: Callable[..., None] = None,
               print_prefix: str = 'Epoch', start_epoch: int = 0,
               validate_interval: int = 10, save: bool = False, amp: bool = False,
               epoch_fn: Callable[..., None] = None,
               get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
               loss_fn: Callable[..., torch.Tensor] = None,
               after_loss_fn: Callable[..., None] = None,
               validate_fn: Callable[..., tuple[float, float]] = None,
               save_fn: Callable[..., None] = None, file_path: str = None, folder_path: str = None, suffix: str = None,
               writer: SummaryWriter = None, main_tag: str = 'train', tag: str = '',
               verbose: bool = True, indent: int = 0,
               adv_train: bool = False, adv_train_alpha: float = 2.0 / 255, adv_train_epsilon: float = 8.0 / 255,
               adv_train_iter: int = 7, **kwargs):
        self.params = optimizer.param_groups[0]['params']
        self.param_numels = [param.data.numel() for param in self.params]
        if after_task_fn is None and hasattr(self, 'after_task_fn'):
            after_task_fn = getattr(self, 'after_task_fn')
        if loader_train is None:
            loader_train = self.dataset.loader['train']
        for task_id in range(self.dataset.task_num):
            self.current_task = task_id
            prints('{green}task{reset}: {0:d}'.format(task_id, **ansi), indent=indent)

            super()._train(loader_train=loader_train[task_id], loader_valid=loader_valid,
                           start_epoch=task_id * epoch, tag=str(task_id), indent=indent + 10,
                           epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler,
                           print_prefix=print_prefix,
                           validate_interval=validate_interval, save=save, amp=amp,
                           epoch_fn=epoch_fn, get_data_fn=get_data_fn, loss_fn=loss_fn, after_loss_fn=after_loss_fn, validate_fn=validate_fn,
                           save_fn=save_fn, file_path=file_path, folder_path=folder_path, suffix=suffix,
                           writer=writer, main_tag=main_tag, verbose=verbose,
                           adv_train=adv_train, adv_train_alpha=adv_train_alpha, adv_train_epsilon=adv_train_epsilon,
                           adv_train_iter=adv_train_iter, **kwargs)
            if callable(after_task_fn):
                after_task_fn(task_id=task_id)
            if isinstance(lr_scheduler, _LRScheduler):
                lr_scheduler.step(0)

    def _validate(self, *args, loader: list[torch.utils.data.DataLoader] = None,
                  _epoch: int = None, print_prefix: str = 'Validate', main_tag: str = 'valid',
                  tag: str = '', writer: SummaryWriter = None, **kwargs):
        losses = SmoothedValue()
        accs = SmoothedValue()
        if loader is None:
            loader = self.dataset.loader['valid']
        for tid in range(self.current_task + 1):
            loader = self.dataset.loader['valid'][tid]
            loss, top1 = super()._validate(*args, loader=loader, tag=str(tid),
                                           print_prefix=print_prefix + str(tid),
                                           _epoch=_epoch, writer=writer,
                                           main_tag=main_tag, **kwargs)
            losses.update(loss)
            accs.update(top1)

        # print("Average Acc: ", np.mean(accs))
        if isinstance(writer, SummaryWriter) and isinstance(_epoch, int) and main_tag:
            writer.add_scalar(f'Acc/{main_tag} average', accs.global_avg, _epoch)
            writer.add_scalar(f'Loss/{main_tag} average', losses.global_avg, _epoch)
        return losses.global_avg, accs.global_avg
