"""Residual CNN submission, adapted from the demo training loop."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    epochs: int = 28
    batch_size: int = 128
    lr: float = 0.006
    weight_decay: float = 0.01
    pct_start: float = 0.2


class ResidualBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels_out),
            )
            if stride != 1 or channels_in != channels_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


def make_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.MaxPool2d(2),
        ResidualBlock(24, 24, 1),
        ResidualBlock(24, 48, 2),
        ResidualBlock(48, 96, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(96, 10),
    ).to(memory_format=torch.channels_last)


def flip_and_turn(x: torch.Tensor) -> torch.Tensor:
    """Apply the demo's independent random flips and transpose on uint8 images."""

    def coin() -> torch.Tensor:
        return torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5

    x = torch.where(coin(), x.flip(3), x)
    x = torch.where(coin(), x.flip(2), x)
    x = torch.where(coin(), x.transpose(2, 3), x)
    return x.contiguous(memory_format=torch.channels_last)


class D4Average(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = None
        for image in (x, x.transpose(2, 3)):
            for dims in ((), (2,), (3,), (2, 3)):
                output = self.core(image.flip(dims) if dims else image)
                scores = output if scores is None else scores + output
        return scores / 8


def train(train_set, log, cfg: Config = Config()) -> nn.Module:
    images, labels = train_set.images, train_set.labels
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=math.ceil(len(images) / cfg.batch_size),
        pct_start=cfg.pct_start,
    )

    for _ in range(cfg.epochs):
        model.train()
        total_loss = torch.zeros(())
        correct = torch.zeros(())
        for batch in torch.randperm(len(images)).split(cfg.batch_size):
            x = flip_and_turn(images[batch]).float().div_(255)
            y = labels[batch]
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total_loss += loss.detach() * len(batch)
            correct += (scores.argmax(1) == y).sum()
        log(train_loss=total_loss.item() / len(images), train_accuracy=correct.item() / len(images))
    return D4Average(model)
