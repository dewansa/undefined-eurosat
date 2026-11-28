"""Demo 2: the leader's accuracy for a quarter of its compute.

A small residual network, trained with SGD for 60 epochs (the first 45 on half-size
images), that answers by averaging over 4 turns of each image. Every setting below was
chosen by an experiment: the notebooks in this folder show which one, and NOTES.md lists
them all. bench.py is the lab version of this file, with everything else we tried.

GitHub calls train(train_set, log), which uses the defaults in Config.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    widths: tuple[int, ...] = (12, 24, 48)  # channels per stage: half the leader's
    epochs: int = 60
    small_epochs: int = 45  # the first epochs see 32x32 images, at a quarter of the cost
    batch_size: int = 64
    lr: float = 0.1  # the peak of the one-cycle schedule
    warmup: float = 0.2  # share of training spent raising the learning rate to its peak
    momentum: float = 0.9
    weight_decay: float = 5e-4
    device: str = "cpu"  # "cuda" or "mps" to try settings faster; GitHub uses the CPU


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions, and a shortcut that carries the input around them."""

    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:  # the input no longer fits: reshape it to match
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def make_network(widths: tuple[int, ...]) -> nn.Sequential:
    """A strided convolution and a max pool shrink 64x64 images to 16x16. Then one residual
    block per stage, each stage after the first at half the size. Then one score per class.
    """
    layers = [
        nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(widths[0]),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ]
    c_in = widths[0]
    for stage, c_out in enumerate(widths):
        layers.append(ResidualBlock(c_in, c_out, stride=1 if stage == 0 else 2))
        c_in = c_out
    layers += [
        nn.AdaptiveAvgPool2d(1),  # average each channel over the whole image
        nn.Flatten(),
        nn.Linear(widths[-1], 10),
    ]
    return nn.Sequential(*layers)


# Turning and mirroring use flips and transposes only: they move numbers without any
# arithmetic, and the leaderboard's FLOP counter knows it.


def random_orientation(x: torch.Tensor) -> torch.Tensor:
    """Give each image in the batch one of its 8 orientations, at random."""

    def coin() -> torch.Tensor:  # True for about half of the images
        return torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5

    x = torch.where(coin(), x.flip(3), x)  # mirror left-right
    x = torch.where(coin(), x.flip(2), x)  # mirror top-bottom
    x = torch.where(coin(), x.transpose(2, 3), x)  # swap rows and columns
    return x.contiguous()  # one tidy block of memory: convolutions run faster on it


def quarter_turns(x: torch.Tensor) -> list[torch.Tensor]:
    """The batch of images turned by 0, 90, 180 and 270 degrees."""
    quarter = x.transpose(2, 3).flip(2)
    return [x, quarter, x.flip(2, 3), quarter.flip(2, 3)]


class AverageOverTurns(nn.Module):
    """Score each image in all 4 quarter turns, and average the scores."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.network(turned) for turned in quarter_turns(x)]).mean(0)


def train(train_set, log, cfg: Config = Config()) -> nn.Module:
    """Train on train_set (uint8 images, 0-255) and return a model that takes float images
    (0-1) and returns 10 class scores per image. log() draws the leaderboard's curves."""
    images = train_set.images.to(cfg.device)
    labels = train_set.labels.to(cfg.device)
    n = len(images)
    network = make_network(cfg.widths).to(cfg.device).train()
    optimizer = torch.optim.SGD(
        network.parameters(),
        cfg.lr,
        momentum=cfg.momentum,
        nesterov=True,
        weight_decay=cfg.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=math.ceil(n / cfg.batch_size),
        pct_start=cfg.warmup,
    )

    for epoch in range(cfg.epochs):
        # Running totals stay on the device: reading one (.item()) makes Python wait for it.
        total_loss = torch.zeros((), device=cfg.device)
        correct = torch.zeros((), device=cfg.device)
        for batch in torch.randperm(n, device=cfg.device).split(cfg.batch_size):
            x = random_orientation(images[batch].float() / 255)
            y = labels[batch]
            if epoch < cfg.small_epochs:
                x = F.avg_pool2d(x, 2)  # each 2x2 square of pixels becomes one
            scores = network(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total_loss += loss.detach() * len(batch)
            correct += (scores.argmax(1) == y).sum()
        log(train_loss=total_loss.item() / n, train_accuracy=correct.item() / n)

    return AverageOverTurns(network).cpu().eval()
