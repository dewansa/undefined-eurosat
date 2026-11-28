"""autoresearch: a small residual CNN, climbed to from a naive CNN one experiment at a time.

How every choice here was made is in NOTES.md and the notebooks. In short:
- batch norm after every convolution, and a learning rate that warms up, then anneals to
  zero (cosine): batch norm learns fast, the annealing makes the end point stable
- each training image is flipped, turned and shifted at random: a satellite photo has no
  "up", and without this the network memorizes
- a first convolution with stride 2 (the 64x64 layer earned nothing), residual blocks
  (the best block the leaderboard's FLOP counter charges fairly), and global average
  pooling into the classifier
- SGD with Nesterov momentum, which beat Adam by a point at equal compute; batches of 128,
  which train 30% faster on a CPU than 64 for the same accuracy
- a small network trained for long: at equal compute it beat every wider or deeper one
- the answer averages the scores of 4 turned copies of the image
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    widths: tuple[int, ...] = (8, 16, 32, 32)  # stem, then one residual block per width
    strides: tuple[int, ...] = (2, 2, 2, 1)
    epochs: int = 40
    batch_size: int = 128  # as accurate as 64 (exp. 15-16), and 30% faster on a CPU
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup: float = 0.1  # share of the steps spent warming up
    shift: int = 4  # random shifts of up to this many pixels
    views: int = 4  # turned copies averaged when answering
    seed: int = 0


def conv_bn(c_in: int, c_out: int, k: int = 3, stride: int = 1, relu: bool = True):
    layers = [nn.Conv2d(c_in, c_out, k, stride, k // 2, bias=False), nn.BatchNorm2d(c_out)]
    return nn.Sequential(*layers, *([nn.ReLU(inplace=True)] if relu else []))


class Residual(nn.Module):
    """Two 3x3 convolutions, added to a shortcut (a 1x1 convolution if the shape changes)."""

    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(c_in, c_out, stride=stride), conv_bn(c_out, c_out, relu=False)
        )
        same = stride == 1 and c_in == c_out
        self.skip = nn.Identity() if same else conv_bn(c_in, c_out, 1, stride, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x))


def make_network(cfg: Config) -> nn.Module:
    stem, *rest = cfg.widths
    layers = [conv_bn(3, stem, stride=cfg.strides[0])]
    for c_in, c_out, stride in zip(cfg.widths, rest, cfg.strides[1:], strict=False):
        layers.append(Residual(c_in, c_out, stride))
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cfg.widths[-1], 10)]
    return nn.Sequential(*layers)


# The 8 symmetries of a square, as (transpose?, dims to flip). The first `views` are used.
SYMMETRIES = [(False, ()), (False, (3,)), (False, (2,)), (False, (2, 3))]
SYMMETRIES += [(True, dims) for _, dims in SYMMETRIES]


class Views(nn.Module):
    """Answer with the average scores of the network over turned copies of the image."""

    def __init__(self, network: nn.Module, views: int) -> None:
        super().__init__()
        self.network = network
        self.views = views

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = 0
        for transpose, dims in SYMMETRIES[: self.views]:
            view = x.transpose(2, 3) if transpose else x
            total = total + self.network(view.flip(dims) if dims else view)
        return total / self.views


def augment(images: torch.Tensor, batch: torch.Tensor, shift: int) -> torch.Tensor:
    """The images of a batch, each flipped, turned and shifted at random, as floats.

    The shift fills in by mirroring the edge. All three are moves of pixels, so they are
    composed into one map of where each output pixel comes from, and done as one gather
    of the uint8 pixels: twice as fast as doing them one after the other.
    """
    n, side = len(batch), images.shape[-1]
    flip_w, flip_h, turn = (torch.rand(n, 1, 1) < 0.5 for _ in range(3))
    dy, dx = torch.randint(0, 2 * shift + 1, (2, n, 1)).unbind(0)
    pos = torch.arange(side) - shift

    def mirror(k: torch.Tensor) -> torch.Tensor:  # positions outside the image reflect in
        return torch.where(k < 0, -k, torch.where(k > side - 1, 2 * (side - 1) - k, k))

    rows = mirror(dy + pos)[:, :, None].expand(n, side, side)
    cols = mirror(dx + pos)[:, None, :].expand(n, side, side)
    rows, cols = torch.where(turn, cols, rows), torch.where(turn, rows, cols)
    rows = torch.where(flip_h, side - 1 - rows, rows)
    cols = torch.where(flip_w, side - 1 - cols, cols)
    where_from = (rows * side + cols).view(n, 1, side * side).expand(n, 3, side * side)
    pixels = images[batch].view(n, 3, side * side).float().div_(255)
    return pixels.gather(2, where_from).view(n, 3, side, side)


def train(train_set, log, cfg: Config = Config()) -> nn.Module:
    torch.manual_seed(cfg.seed)
    images, labels = train_set.images, train_set.labels
    network = make_network(cfg)
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        nesterov=True,
        weight_decay=cfg.weight_decay,
    )
    total = math.ceil(len(images) / cfg.batch_size) * cfg.epochs
    warm = round(cfg.warmup * total)

    def factor(step: int) -> float:  # linear warm-up, then cosine down to zero
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    for _ in range(cfg.epochs):
        network.train()
        loss_sum = torch.zeros(())
        correct = torch.zeros(())
        for batch in torch.randperm(len(images)).split(cfg.batch_size):
            if len(batch) < 2:  # batch norm needs two images
                continue
            x = augment(images, batch, cfg.shift)
            y = labels[batch]
            scores = network(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            loss_sum += loss.detach() * len(batch)
            correct += (scores.argmax(1) == y).sum()
        log(
            train_loss=loss_sum.item() / len(images),
            train_accuracy=correct.item() / len(images),
        )
    return Views(network.eval(), cfg.views)
