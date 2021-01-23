#!/usr/bin/env python3

from .split_model import SplitModel
from trojanzoo.utils.data import dataset_to_list, sample_batch
from trojanzoo.utils.influence import InfluenceFunction
from trojanzoo.environ import env

import torch
from kmeans_pytorch import kmeans
import numpy as np
import quadprog

from typing import TYPE_CHECKING
import argparse    # TODO: python 3.10
if TYPE_CHECKING:
    import torch.nn
    import torch.autograd
    import torch.optim
    import torch.utils.data


class GEM(SplitModel):
    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--memory_size', dest='memory_size', type=int)
        group.add_argument('--memory_method', dest='memory_method')

    def __init__(self, *args, memory_size: int = 256, memory_method: str = 'random', **kwargs):
        super().__init__(*args, **kwargs)
        self.param_list['gem'] = ['memory_size', 'memory_method']
        self.memory_size = memory_size
        self.memory_method = memory_method
        self.memory_data: list[torch.Tensor] = []
        self.memory_targets: list[torch.Tensor] = []
        self.influence = InfluenceFunction(model=self)
        self.memory_hess: list[torch.Tensor] = []

    def after_task_fn(self, task_id: int):
        if self.memory_method == 'influence':
            return
        dataset = self.dataset.loader['train'][task_id].dataset
        data, targets = self.sample_batch(dataset, batch_size=self.memory_size)
        self.memory_data.append(torch.stack(data).pin_memory())
        self.memory_targets.append(torch.tensor(targets, dtype=torch.long).pin_memory())

    def epoch_fn(self, optimizer: torch.optim.Optimizer, _epoch: int = None, epoch: int = None,
                 start_epoch: int = None, **kwargs):
        if self.memory_method != 'influence' or _epoch != 2:
            return
        task_id = self.current_task
        dataset = self.dataset.loader['train'][task_id].dataset
        data, targets = self.sample_batch(dataset, batch_size=self.memory_size)
        self.memory_data.append(torch.stack(data).pin_memory())
        self.memory_targets.append(torch.tensor(targets, dtype=torch.long).pin_memory())
        # if not hasattr(self, 'optimizer'):
        #     self.base_lr: float = optimizer.param_groups[0]['lr']
        #     self.optimizer = optimizer
        # lr = self.base_lr * (1 - (_epoch + start_epoch) / (epoch * self.dataset.split_num))
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = lr

    def sample_batch(self, dataset: torch.utils.data.Dataset, batch_size: int,
                     memory_method: str = None) -> tuple[list, list[int]]:
        memory_method = memory_method if memory_method is not None else self.memory_method

        _, targets = dataset_to_list(dataset, label_only=True)
        class_list = list(set(targets))
        _input = []
        _label: list[int] = []
        for _class in class_list:
            subset = self.dataset.get_class_set(dataset, [_class])
            # memory_method_fn: Callable[[torch.utils.data.Dataset, int], (list[torch.Tensor], list[int])] = sample_batch
            current_input, current_label = self.sample_batch_fn(
                subset, batch_size=batch_size, memory_method=memory_method)
            _input.extend(current_input)
            _label.extend(current_label)
        return _input, _label

    def sample_batch_fn(self, dataset: torch.utils.data.Dataset, batch_size: int,
                        memory_method: str = None) -> tuple[list, list[int]]:
        if memory_method == 'influence':
            return self.sample_batch_influence(dataset, batch_size=batch_size)
        elif memory_method == 'cluster':
            return self.sample_batch_cluster(dataset, batch_size=batch_size)
        elif memory_method == 'distillation':
            return self.sample_batch_distillation(dataset, batch_size=batch_size)
        return sample_batch(dataset, batch_size=batch_size)

    def sample_batch_distillation(self, dataset: torch.utils.data.Dataset, batch_size: int) -> tuple[list, list[int]]:
        model_copy = self.state_dict()

        synthesize_data = torch.rand()

        self.load_state_dict(model_copy)
        return None

    def sample_batch_cluster(self, dataset: torch.utils.data.Dataset, batch_size: int) -> tuple[list, list[int]]:
        _, targets = dataset_to_list(dataset, label_only=True)  # list[int]
        _labels = torch.tensor(targets, device=env['device'])
        num_clusters = len(set(targets)) * 2
        loader = self.dataset.get_dataloader(dataset=dataset, shuffle=False)
        feats: list[torch.Tensor] = []
        for data in loader:
            _input, _label = self.get_data(data)
            # feats.append(self.influence.calc_v(_input, _label).detach())
            feats.append(self.get_prob(_input).detach())
        feats = torch.cat(feats)    # (N, D)

        pred_results: list[torch.Tensor] = []
        for data in loader:
            _input, _ = self.get_data(data)
            pred_results.append(self.get_class(_input).detach())
        pred_results = torch.cat(pred_results)    # (N, num_classes)
        truth_idx = torch.arange(len(_labels))[_labels == pred_results]     # (N')
        idx: list[int] = []
        if len(truth_idx):
            # cluster_labels (N'), cluster_centers (num_clusters, D)
            cluster_labels, cluster_centers = kmeans(X=feats[truth_idx], num_clusters=num_clusters,
                                                     distance='euclidean', device=env['device'])
            cluster_centers: torch.Tensor = cluster_centers.to(env['device'])
            correct_n = 0
            remain_n = batch_size
            for c in range(num_clusters):
                # majority vote
                cluster_idx = truth_idx[cluster_labels == c]    # (N'_c)
                truth_labels = _labels[cluster_idx]   # (N'_c)
                truth_major_label = int(truth_labels.mode()[0])
                correct_idx = cluster_idx[truth_labels == truth_major_label]    # (N'_c')
                correct_n += len(correct_idx)

                # calculate dists
                center: torch.Tensor = cluster_centers[c]
                dists = torch.cdist(feats[correct_idx], center.unsqueeze(dim=0)).flatten()  # (N'_c')
                sorted_idx = correct_idx[torch.argsort(dists)]

                if c != num_clusters - 1:
                    idx.extend(sorted_idx.tolist()[:(remain_n // (num_clusters - c))])
                else:
                    idx.extend(sorted_idx.tolist()[:remain_n])
                remain_n -= (remain_n // (num_clusters - c))
            print(f'\nKmeans clustering accuracy: {correct_n / len(truth_idx):.4f}')
        else:
            truth_idx = torch.arange(len(_labels))

        if len(idx) < batch_size:
            rest_idx = list(set(truth_idx) - set(idx))
            if len(rest_idx) < batch_size - len(idx):
                rest_idx = list(set(torch.arange(len(_labels))) - set(idx))
            idx += list(np.random.choice(rest_idx, batch_size - len(idx), replace=False))
        return sample_batch(dataset, idx=idx)

    def sample_batch_influence(self, dataset: torch.utils.data.Dataset, batch_size: int,
                               orthogonal: bool = False) -> tuple[list, list[int]]:
        assert len(dataset) >= batch_size
        loader = self.dataset.get_dataloader(dataset=dataset, shuffle=False)
        hess_inv = torch.cholesky_inverse(self.influence.calc_H(loader))
        self.memory_hess.append(hess_inv.detach().cpu())
        idx: list[int] = []
        if orthogonal:
            memory_bases: torch.Tensor = torch.zeros(0, self.influence.parameter.numel(),
                                                     device=env['device'])  # (M, D)
            for _ in range(batch_size):
                influence_list: list[float] = []
                v_list: torch.Tensor = torch.zeros(0, memory_bases.shape[1],
                                                   device=memory_bases.device)  # (0, D)
                memory_metric = hess_inv @ memory_bases.t() @ memory_bases  # (D,D)
                for data in loader:
                    _input, _label = self.get_data(data, adv=True)
                    v = self.influence.calc_v(_input, _label)   # (N, D)
                    v -= v @ memory_metric   # (N, D)
                    v_list = torch.cat([v_list, v])
                    influence_list.extend(self.influence.up_loss(v=v, hess_inv=hess_inv))
                    # Gram–Schmidt orthogonalization
                    # for j in range(len(v)):
                    #     for i in range(len(memory_bases)):
                    #         v[j] -= v[j] @ memory_bases[i] * memory_bases[i]
                    # # v -= v @ memory_bases.t() @ memory_bases
                    # memory_bases are already normalized
                best_idx = int(np.argmax(influence_list))
                memory_bases = torch.cat([memory_bases, v_list[best_idx].unsqueeze(0) / influence_list[best_idx]])
                idx.append(best_idx)
        else:
            influence_list: list[float] = []
            for data in loader:
                _input, _label = self.get_data(data, adv=True)
                influence_list.extend(self.influence.up_loss(_input, _label, hess_inv=hess_inv))
            idx = list(range(len(dataset)))
            _, idx = zip(*sorted(zip(influence_list, idx)))
            idx = list(idx)[-batch_size:]
        return sample_batch(dataset, idx=idx)

    def after_loss_fn(self, optimizer: torch.optim.Optimizer,
                      _iter: int = None, total_iter: int = None, **kwargs):
        # self.lr_decay_fn(optimizer=optimizer, _iter=_iter, total_iter=total_iter)
        if not hasattr(self, 'params'):
            self.params: list[torch.nn.Parameter] = [param for param in self.parameters() if param.requires_grad]
            self.grad_dims: list[int] = [param.data.numel() for param in self.params]
        if self.current_task > 0:
            current_grad, prev_grad = self.store_grad()
            grad = self.project2cone2(current_grad, prev_grad)
            self.rewrite_grad(grad)

    def lr_decay_fn(self, optimizer: torch.optim.Optimizer,
                    # start_epoch: int = None, _epoch: int = None, epoch: int = None,
                    _iter: int = None, total_iter: int = None):
        if not hasattr(self, 'optimizer'):
            self.base_lr: float = optimizer.param_groups[0]['lr']
            self.optimizer = optimizer
        lr = self.base_lr * (1 - (_iter + self.current_task * total_iter) / (self.dataset.split_num * total_iter))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def rewrite_grad(self, grad: torch.Tensor):
        split_grads: tuple[torch.Tensor] = torch.split(grad, self.grad_dims)
        for param, split_grad in zip(self.params, split_grads):
            param.grad.data.copy_(split_grad.to(param.device).view_as(param))

    def store_grad(self) -> tuple[torch.Tensor, torch.Tensor]:
        current_grad = torch.cat([param.grad.flatten() for param in self.params])
        self.zero_grad()

        prev_grad = []
        for task_id in range(self.current_task):
            memory_data = self.memory_data[task_id].to(device=env['device'])
            memory_targets = self.memory_targets[task_id].to(device=env['device'])
            memory_loss = self.loss(_input=memory_data, _label=memory_targets)
            memory_loss.backward()
            prev_grad.append(torch.cat([param.grad.flatten() for param in self.params]))
            self.zero_grad()
        prev_grad = torch.stack(prev_grad)
        return current_grad, prev_grad

    @staticmethod
    def project2cone2(current_grad: torch.Tensor, prev_grad: torch.Tensor,
                      margin: float = 0.5, eps: float = 1e-3) -> torch.Tensor:
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.
            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector
        """
        if (prev_grad @ current_grad < 0).sum() == 0:
            return current_grad
        device = current_grad.device
        with torch.no_grad():
            current_grad = current_grad.detach().cpu()  # (p)
            prev_grad = prev_grad.detach().cpu()    # (t, p)
            t = prev_grad.shape[0]
            P = prev_grad @ prev_grad.t()  # (t, t)
            P = (P + P.t()) / 2 + torch.eye(t) * eps  # keep it positive-definitive
            q = -prev_grad @ current_grad  # (t)
            P = P.double().numpy()
            q = q.double().numpy()
            G = np.eye(t)   # (t, t)
            h = margin * np.ones(t)  # (t)
            v = torch.as_tensor(quadprog.solve_qp(P, q, G, h)[0]).float()   # (t)
            x = v @ prev_grad + current_grad   # (t)
            return x.to(device=device)
