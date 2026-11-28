"""Demo 2: as accurate as the top of the leaderboard, for far less compute.

How it was built, and why each setting has its value: the notebooks in this folder.
Notebook 3's submission cell writes this file, so make your changes there.

GitHub calls train(train_set, log), which uses the defaults in Config.
The notebooks call train(train_set, log, Config(epochs=8)) to try other values.
"""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)  # frozen: a setting cannot be changed by accident
class Config:
    # The network
    stem: str = "conv"  # first layer: "conv" (3x3 windows) or "patch" (cut into squares)
    stem_stride: int = 2  # "conv": move this many pixels at a time; "patch": square size
    stem_pool: bool = True  # "conv" only: halve the image again with a max pool
    block: str = "basic"  # "basic": two 3x3 convolutions; "inverted": cheap depthwise ones
    widths: tuple[int, ...] = (24, 48, 96)  # channels in each stage
    depths: tuple[int, ...] = (1, 1, 1)  # blocks per stage; each stage after the first halves
    expand: int = 4  # "inverted" only: how much wider the inside of a block is
    zero_init: bool = False  # start each block as "do nothing" (its last scale at 0)
    # Training
    optimizer: str = "adamw"  # "adamw", or "sgd" (with Nesterov momentum 0.9)
    epochs: int = 28  # passes over all training images
    batch_size: int = 128  # images per learning step
    lr: float = 0.006  # learning rate: the largest step size
    weight_decay: float = 0.01  # pull weights toward 0
    pct_start: float = 0.2  # share of training spent raising the learning rate
    label_smoothing: float = 0.0  # aim for 0.9 instead of 1 on the right class (with 0.1)
    mixup: float = 0.0  # blend pairs of images and their labels (0: off; 0.2: mild)
    ema: float = 0.0  # keep a running average of the weights, and return that (0: off)
    augment: bool = True  # randomly turn and mirror images
    low_res: float = 0.0  # share of epochs trained on half-size (32x32) images: 4x cheaper
    # Answering
    tta: int = 8  # look at the image in 1, 2, 4 or 8 orientations and average the answers
    # Watching
    valid_images: int = 0  # kept out of training, only to draw the validation curve
    device: str = "cpu"  # notebooks only: "cuda" or "mps" trains on a GPU (GitHub has none)
    watch: Callable | None = None  # notebooks only: gets (model, optimizer) to look inside


# ---------------------------------------------------------------- the network


class Basic(nn.Module):
    """Two 3x3 convolutions and a shortcut around them (a ResNet "basic block")."""

    def __init__(self, c_in: int, c_out: int, stride: int, cfg: Config) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.skip = shortcut(c_in, c_out, stride)
        if cfg.zero_init:
            nn.init.zeros_(self.body[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x))


class Inverted(nn.Module):
    """Widen with a 1x1 convolution, look around with a cheap depthwise 3x3, narrow again.

    A depthwise 3x3 convolution looks at each channel on its own: 9 multiplies per
    channel and pixel instead of 9 x channels. The 1x1 convolutions mix the channels.
    This is the block of MobileNetV2.
    """

    def __init__(self, c_in: int, c_out: int, stride: int, cfg: Config) -> None:
        super().__init__()
        wide = c_in * cfg.expand
        self.body = nn.Sequential(
            nn.Conv2d(c_in, wide, 1, bias=False),
            nn.BatchNorm2d(wide),
            nn.ReLU(),
            nn.Conv2d(wide, wide, 3, stride, 1, groups=wide, bias=False),  # depthwise
            nn.BatchNorm2d(wide),
            nn.ReLU(),
            nn.Conv2d(wide, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.skip = shortcut(c_in, c_out, stride)
        if cfg.zero_init:
            nn.init.zeros_(self.body[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


def shortcut(c_in: int, c_out: int, stride: int) -> nn.Module:
    """The way around a block: nothing to do, or a 1x1 convolution when the shape changes."""
    if stride == 1 and c_in == c_out:
        return nn.Identity()
    return nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))


def make_model(cfg: Config = Config()) -> nn.Sequential:
    w = cfg.widths[0]
    if cfg.stem == "patch":  # cut the image into squares and describe each one
        stem = [nn.Conv2d(3, w, cfg.stem_stride, cfg.stem_stride, bias=False)]
    else:
        stem = [nn.Conv2d(3, w, 3, cfg.stem_stride, 1, bias=False)]
    stem += [nn.BatchNorm2d(w), nn.ReLU()]
    if cfg.stem == "conv" and cfg.stem_pool:
        stem.append(nn.MaxPool2d(2))

    block = {"basic": Basic, "inverted": Inverted}[cfg.block]
    blocks, c_in = [], w
    for stage, (c_out, depth) in enumerate(zip(cfg.widths, cfg.depths, strict=True)):
        for i in range(depth):
            stride = 2 if stage > 0 and i == 0 else 1  # each new stage halves the size
            blocks.append(block(c_in, c_out, stride, cfg))
            c_in = c_out
    return nn.Sequential(
        *stem,
        *blocks,
        nn.AdaptiveAvgPool2d(1),  # average each pattern over the whole image
        nn.Flatten(),
        nn.Linear(c_in, 10),  # one score per class
    )


# The 4 quarter turns of an image, and each of them mirrored: 8 orientations.
# The first 2 (and the first 4) are a group on their own: turning by half (or by a
# quarter) again gives one of them, so averaging over them treats each alike.
def turn(x: torch.Tensor, k: int) -> torch.Tensor:
    x = torch.rot90(x, k % 4, dims=(2, 3))
    return x.flip(3) if k >= 4 else x


ORIENTATIONS = (0, 2, 1, 3, 4, 6, 5, 7)


class Average(nn.Module):
    """Answer for several orientations of each image and average the scores."""

    def __init__(self, core: nn.Module, views: int) -> None:
        super().__init__()
        self.core, self.views = core, views

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.views == 1:
            return self.core(x)
        return sum(self.core(turn(x, k)) for k in ORIENTATIONS[: self.views]) / self.views


# ---------------------------------------------------------------- training


def coin(x: torch.Tensor) -> torch.Tensor:
    """True for about half of the images in the batch x, at random."""
    return torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5


def flip_and_turn(x: torch.Tensor) -> torch.Tensor:
    """Give each image one of its 8 orientations, at random."""
    x = torch.where(coin(x), x.flip(3), x)  # mirror left-right
    x = torch.where(coin(x), x.flip(2), x)  # mirror top-bottom
    x = torch.where(coin(x), x.transpose(2, 3), x)  # swap rows and columns
    return x.contiguous()


def update_average(average: nn.Module, model: nn.Module, decay: float) -> None:
    """Move the averaged weights a little toward the current ones (cheap: no FLOPs)."""
    with torch.no_grad():
        mine = list(average.state_dict().values())
        theirs = list(model.state_dict().values())
        floats = [(a, b) for a, b in zip(mine, theirs, strict=True) if a.is_floating_point()]
        torch._foreach_lerp_([a for a, _ in floats], [b for _, b in floats], 1 - decay)
        for a, b in zip(mine, theirs, strict=True):
            if not a.is_floating_point():
                a.copy_(b)


def train(train_set, log, cfg: Config = Config()) -> nn.Module:
    images, labels = train_set.images.to(cfg.device), train_set.labels.to(cfg.device)

    # Keep a few images out of training, only to draw the validation curve.
    order = torch.randperm(len(images), device=cfg.device)
    valid, fit = order[: cfg.valid_images], order[cfg.valid_images :]
    if cfg.valid_images:
        log.validate_on(images[valid].cpu(), labels[valid].cpu())
    images, labels = images[fit], labels[fit]

    model = make_model(cfg).to(cfg.device)
    average = copy.deepcopy(model) if cfg.ema else None
    if cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(images) / cfg.batch_size)
    # One cycle: the step size grows to lr, then shrinks to almost nothing by the end.
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=cfg.pct_start,
    )
    if cfg.watch:
        cfg.watch(model, optimizer)

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        small = epoch < round(cfg.low_res * cfg.epochs)  # the first epochs on 32x32 images
        total_loss = torch.zeros((), device=cfg.device)
        correct = torch.zeros((), device=cfg.device)
        for batch in torch.randperm(len(images), device=cfg.device).split(cfg.batch_size):
            x = images[batch].float() / 255  # bytes 0-255 become decimals 0-1
            y = labels[batch]
            if cfg.augment:
                x = flip_and_turn(x)
            if small:
                x = F.avg_pool2d(x, 2)  # each 2x2 square of pixels becomes one
            if cfg.mixup:  # each image becomes a blend of itself and another one
                share = torch.distributions.Beta(cfg.mixup, cfg.mixup).sample().item()
                other = torch.randperm(len(x), device=x.device)
                x = share * x + (1 - share) * x[other]
            scores = model(x)
            loss = F.cross_entropy(scores, y, label_smoothing=cfg.label_smoothing)
            if cfg.mixup:  # ...and should answer with the same blend of the two labels
                other_loss = F.cross_entropy(scores, y[other], label_smoothing=cfg.label_smoothing)
                loss = share * loss + (1 - share) * other_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            step += 1
            if average is not None:  # average over about the last 1/(1-ema) steps
                update_average(average, model, min(cfg.ema, (1 + step) / (10 + step)))
            total_loss += loss.detach() * len(batch)
            correct += (scores.argmax(1) == y).sum()
        n = len(images)
        final = model if average is None else average
        # log(model): GitHub checks the model on the validation images later, off the clock
        log(final, train_loss=total_loss.item() / n, train_accuracy=correct.item() / n)
    return Average(final, cfg.tta).cpu().eval()
