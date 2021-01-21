#!/usr/bin/env python3

from trojanvision.models import ImageModel
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils.output import prints, ansi

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from typing import TYPE_CHECKING
from typing import Callable
from lifelong.datasets.split_dataset import SplitDataset    # TODO: python 3.10
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
if TYPE_CHECKING:
    import torch.utils.data


class SplitModel(ImageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset: SplitDataset
        self.current_task: int = 0
        self.task_mask = torch.zeros(self.dataset.split_num, self.num_classes, dtype=torch.bool)
        for task_id in range(self.dataset.split_num):
            self.task_mask[task_id][torch.tensor(self.dataset.class_order_list[task_id])] = True
        label_to_task: dict[int, int] = {
            label: task for task, label_list in enumerate(self.dataset.class_order_list)
            for label in label_list}
        self.label_to_task = torch.tensor([label_to_task[label] for label in range(self.num_classes)])
        self.label_to_idx = torch.tensor(
            [sorted(self.dataset.class_order_list[self.label_to_task[label]]).index(label) for label in range(self.num_classes)])

    def loss(self, _input: torch.Tensor = None, _label: torch.Tensor = None,
             _output: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if _output is None:
            _output = self(_input, **kwargs)
        _task = self.label_to_task[_label]
        _mask = self.task_mask[_task]
        _output[~_mask] = -1e10
        _output = _output[_mask].view(len(_input), -1)
        _label = self.label_to_idx[_label].to(device=_label.device)
        return super().loss(_input=_input, _label=_label, _output=_output, **kwargs)

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
        if after_task_fn is None and hasattr(self, 'after_task_fn'):
            after_task_fn = getattr(self, 'after_task_fn')
        if loader_train is None:
            loader_train = self.dataset.loader['train']
        for task_id in range(self.dataset.split_num):
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
