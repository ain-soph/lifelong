#!/usr/bin/env python3

from trojanzoo.environ import env
import torch
import torch.autograd
import torch.utils.data
import torch.optim
from torch import nn
from torch.nn import functional as F

EPSILON = 1e-16


class WGAN:
    def __init__(self, data_shape: list[int], z_size: int = 100, g_channel_size: int = 64, c_channel_size: int = 64,
                 gp_lambda: float = 10., importance_of_new_task: float = 0.4):
        image_channel_size = data_shape[0]
        image_size = data_shape[1] * data_shape[2]
        self.generator = Generator(image_size=image_size, image_channel_size=image_channel_size,
                                   z_size=z_size, channel_size=g_channel_size)
        self.critic = Critic(image_size=image_size, image_channel_size=image_channel_size, channel_size=c_channel_size)
        self.gp_lambda = gp_lambda
        self.importance_of_new_task = importance_of_new_task

    def _train(self, epoch: int, n_critic: int, loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
               prev_loader: torch.utils.data.DataLoader = None):
        if prev_loader is not None:
            prev_loader = iter(prev_loader)
        prev_input: torch.Tensor = None
        for _epoch in range(epoch):
            for data in loader:
                _input, _ = self.get_data(data)
                if prev_loader is not None:
                    prev_input, _ = self.get_data(next(prev_loader))
                for _ in range(n_critic):
                    z = 0.1 * torch.randn(_input.shape[0], self.generator.z_size)
                    loss = self.loss(_input, z=z)
                    if prev_loader is not None:
                        prev_loss = self.loss(prev_input, z=z)
                        loss = self.importance_of_new_task * loss + (1 - self.importance_of_new_task) * prev_loss
                    loss.backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

    def loss(self, _input: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        z = z if z is not None else 0.1 * torch.randn(_input.shape[0], self.generator.z_size)
        g_input: torch.Tensor = self.generator(z)
        c_g: torch.Tensor = self.critic(g_input)
        c_x: torch.Tensor = self.critic(_input)

        eps = torch.rand(_input.shape[0])
        x_hat: torch.Tensor = (eps * _input + (1 - eps) * g_input).detach()
        grad = torch.autograd.grad(self.critic(x_hat), x_hat, create_graph=True, retain_graph=True)[0]
        gp_penalty: torch.Tensor = self.gp_lambda * (1 - (grad + EPSILON).norm(p=2, dim=1))**2
        return (c_g - c_x + gp_penalty).mean()

    @staticmethod
    def get_data(data: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return data[0].to(env['device'], non_blocking=True), data[1].to(env['device'], dtype=torch.long, non_blocking=True)


class Generator(nn.Module):
    def __init__(self, image_size: int, image_channel_size: int, z_size: int = 100, channel_size: int = 64):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.fc = nn.Linear(z_size, (image_size // 8)**2 * channel_size * 8)
        self.bn0 = nn.BatchNorm2d(channel_size * 8)
        self.bn1 = nn.BatchNorm2d(channel_size * 4)
        self.deconv1 = nn.ConvTranspose2d(
            channel_size * 8, channel_size * 4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        self.deconv2 = nn.ConvTranspose2d(
            channel_size * 4, channel_size * 2,
            kernel_size=4, stride=2, padding=1,
        )
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv3 = nn.ConvTranspose2d(
            channel_size * 2, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            channel_size, image_channel_size,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        g = F.relu(self.bn0(self.fc(z).view(
            z.size(0),
            self.channel_size * 8,
            self.image_size // 8,
            self.image_size // 8,
        )))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return F.sigmoid(g)


class Critic(nn.Module):
    def __init__(self, image_size: int, image_channel_size: int, channel_size: int = 64):
        # configurations
        super().__init__()
        self.image_size = image_size    # H*W
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            channel_size, channel_size * 2,
            kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            channel_size * 2, channel_size * 4,
            kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            channel_size * 4, channel_size * 8,
            kernel_size=4, stride=1, padding=1,
        )
        self.fc = nn.Linear((image_size // 8)**2 * channel_size * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, (self.image_size // 8)**2 * self.channel_size * 4)
        return self.fc(x)
