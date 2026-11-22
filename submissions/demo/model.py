"""Demo: a small convolutional network, built from the baseline step by step.

How it was built, and why each setting has its value: 2-build-model.ipynb.
That notebook's submission cell writes this file, so make your changes there.

GitHub calls train(train_set, log), which uses the defaults in Config.
The notebook calls train(train_set, log, Config(lr=0.03)) to try other values.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)  # frozen: a setting cannot be changed by accident
class Config:
    widths: tuple[int, ...] = (16, 24, 32, 48)  # how many patterns each block looks for
    stem_stride: int = 2  # the first block moves 2 pixels at a time: 4x less work
    batch_norm: bool = True  # rescale each block's outputs while learning
    epochs: int = 11  # passes over all training images
    batch_size: int = 64  # images per learning step
    lr: float = 0.01  # learning rate: the largest step size
    weight_decay: float = 0.0  # pull weights toward 0 (off: it did not help here)
    augment: bool = True  # randomly turn and mirror images
    valid_images: int = 1000  # kept out of training, only to draw the validation curve
    device: str = "cpu"  # notebook only: "mps" trains on a Mac's GPU (GitHub has none)
    watch: Callable | None = None  # notebook only: gets (model, optimizer) to look inside


def block(channels_in: int, channels_out: int, stride: int, batch_norm: bool) -> nn.Sequential:
    """Find patterns, rescale them, keep the positive ones, halve the image size."""
    return nn.Sequential(
        # 3x3 windows; no bias with batch norm, which adds its own shift
        nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=not batch_norm),
        nn.BatchNorm2d(channels_out) if batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def make_model(cfg: Config = Config()) -> nn.Sequential:
    sizes = [3, *cfg.widths]  # 3 colors come in; each block's output feeds the next one
    blocks = [
        block(c_in, c_out, cfg.stem_stride if i == 0 else 1, cfg.batch_norm)
        for i, (c_in, c_out) in enumerate(pairwise(sizes))  # (3, 16), (16, 32), ...
    ]
    return nn.Sequential(
        *blocks,
        nn.AdaptiveAvgPool2d(1),  # average each pattern over the whole image
        nn.Flatten(),
        nn.Linear(cfg.widths[-1], 10),  # one score per class
    )


def coin(x: torch.Tensor) -> torch.Tensor:
    """True for about half of the images in the batch x, at random."""
    return torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5


def flip_and_turn(x: torch.Tensor) -> torch.Tensor:
    """Give each image one of its 8 versions: 4 quarter turns, each mirrored or not."""
    x = torch.where(coin(x), x.flip(3), x)  # mirror left-right
    x = torch.where(coin(x), x.flip(2), x)  # mirror top-bottom
    x = torch.where(coin(x), x.transpose(2, 3), x)  # swap rows and columns
    return x.contiguous()  # tidy the memory layout: a Mac GPU is 5x slower without it


def train(train_set, log, cfg: Config = Config()) -> nn.Module:
    images, labels = train_set.images.to(cfg.device), train_set.labels.to(cfg.device)

    # Keep a few images out of training, only to draw the validation curve.
    order = torch.randperm(len(images), device=cfg.device)
    valid, fit = order[: cfg.valid_images], order[cfg.valid_images :]
    log.validate_on(images[valid].cpu(), labels[valid].cpu())
    images, labels = images[fit], labels[fit]

    model = make_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # One cycle: the step size grows to lr, then shrinks to almost nothing by the end.
    steps_per_epoch = math.ceil(len(images) / cfg.batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg.lr, epochs=cfg.epochs, steps_per_epoch=steps_per_epoch
    )
    if cfg.watch:
        cfg.watch(model, optimizer)

    for _ in range(cfg.epochs):
        model.train()
        # Totals stay on the device: reading a number back (.item()) makes Python wait
        # for the device, so we do that once per epoch instead of after every step.
        total_loss = torch.zeros((), device=cfg.device)
        correct = torch.zeros((), device=cfg.device)
        for batch in torch.randperm(len(images), device=cfg.device).split(cfg.batch_size):
            x = images[batch].float() / 255  # bytes 0-255 become decimals 0-1
            y = labels[batch]
            if cfg.augment:
                x = flip_and_turn(x)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total_loss += loss.detach() * len(batch)
            correct += (scores.argmax(1) == y).sum()
        # log(model): GitHub checks the model on the validation images later, off the clock
        n = len(images)
        log(model, train_loss=total_loss.item() / n, train_accuracy=correct.item() / n)
    return model
