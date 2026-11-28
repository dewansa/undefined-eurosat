"""The lab bench for autoresearch: one network you describe in a line, one training loop.

Every experiment is a `Config`. The network is written as a short string, read left to
right, one layer (or block) per word:

    "c16 p c32 p c64 p fl"     the naive CNN: conv 16, max pool, conv 32, pool, conv 64,
                               pool, flatten, then a linear layer to the 10 classes

    c<C>      3x3 convolution to C channels, then ReLU (and batch norm if cfg.norm)
              suffixes: k<K> kernel size, s<S> stride.  c32k5s2 = 5x5, stride 2
    p         2x2 max pool (halves the image)
    r<C>      residual block: two 3x3 convolutions and a shortcut (He et al., 2015)
    d<C>      depthwise-separable block: a 3x3 conv per channel, then a 1x1 conv
              that mixes channels (MobileNet, Howard et al., 2017)
    m<C>      inverted residual: 1x1 expand (e<E>, default 4), 3x3 per channel, 1x1
              project, shortcut when the shape allows (MobileNetV2, Sandler et al., 2018)
    fl        flatten          gap   global average pool (one number per channel)

The head (a linear layer to 10 classes) is added at the end.

Run experiments from experiments.py, several at once on one GPU:

    python search.py 0-naive 1-lr --device cuda --workers 8

Each finished run is one line of runs/<experiment>.jsonl. Runs already there are skipped,
so the same command can be run again after adding more settings.

Numbers carry over to a CPU: the GPU's TF32 shortcut is turned off, so it does float32
arithmetic like a CPU does. The FLOPs are counted with the repo's own counter
(eurosat.flops), on one training step, so they match the leaderboard's count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))  # the pod runs without installing the repo

from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from eurosat import data  # noqa: E402
from eurosat.flops import FORMULAS, counting  # noqa: E402

RUNS = Path(os.environ.get("AUTORESEARCH_RUNS", HERE / "runs"))  # the pod keeps its own


@dataclass(frozen=True)
class Config:
    arch: str = "c16 p c32 p c64 p fl"
    norm: bool = False  # batch norm after every convolution
    standardize: bool = False  # subtract the mean colour, divide by its spread (in the model)
    epochs: int = 10
    batch: int = 64
    opt: str = "adam"  # adam, adamw or sgd (Nesterov momentum 0.9)
    lr: float = 1e-3
    wd: float = 0.0  # weight decay
    sched: str = "const"  # const, or cos: linear warm-up, then cosine down to zero
    warmup: float = 0.0  # share of the steps spent warming up (cos only)
    aug: str = "none"  # none, or any of hflip / flips / d4, plus "+crop<P>" for shifts
    smooth: float = 0.0  # label smoothing
    tta: int = 1  # views averaged when answering: 1, 2, 4 or 8


# ---------------------------------------------------------------- the network


def conv_bn(cin: int, cout: int, k: int = 3, s: int = 1, groups: int = 1, norm=True, act=True):
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, s: int, norm: bool) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(cin, cout, k, s, norm=norm), conv_bn(cout, cout, k, norm=norm, act=False)
        )
        same = s == 1 and cin == cout
        self.skip = nn.Identity() if same else conv_bn(cin, cout, 1, s, norm=norm, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x))


def separable(cin: int, cout: int, k: int, s: int, norm: bool) -> nn.Module:
    return nn.Sequential(
        conv_bn(cin, cin, k, s, groups=cin, norm=norm), conv_bn(cin, cout, 1, norm=norm)
    )


class Inverted(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, s: int, e: int, norm: bool) -> None:
        super().__init__()
        mid = cin * e
        self.body = nn.Sequential(
            conv_bn(cin, mid, 1, norm=norm),
            conv_bn(mid, mid, k, s, groups=mid, norm=norm),
            conv_bn(mid, cout, 1, norm=norm, act=False),
        )
        self.shortcut = s == 1 and cin == cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) if self.shortcut else self.body(x)


TOKEN = re.compile(r"^([a-z]+?)(\d*)((?:[kse]\d+)*)$")


def parse(word: str) -> tuple[str, int, dict[str, int]]:
    match = TOKEN.match(word)
    if not match:
        raise ValueError(f"cannot read the layer {word!r}")
    kind, channels, options = match.groups()
    opts = {key: int(value) for key, value in re.findall(r"([kse])(\d+)", options)}
    return kind, int(channels or 0), opts


class Net(nn.Module):
    """The layers the arch string names, a linear head, and optional input standardizing."""

    def __init__(self, cfg: Config, mean=None, std=None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c, side = 3, 64
        for word in cfg.arch.split():
            kind, out, o = parse(word)
            k, s = o.get("k", 3), o.get("s", 1)
            if kind == "c":
                layers.append(conv_bn(c, out, k, s, norm=cfg.norm))
            elif kind == "r":
                layers.append(Residual(c, out, k, s, cfg.norm))
            elif kind == "d":
                layers.append(separable(c, out, k, s, cfg.norm))
            elif kind == "m":
                layers.append(Inverted(c, out, k, s, o.get("e", 4), cfg.norm))
            elif kind == "p":
                layers.append(nn.MaxPool2d(2))
                s = 2
            elif kind == "gap":
                layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
                side = 1
            elif kind == "fl":
                layers.append(nn.Flatten())
                c, side = c * side * side, 1
            else:
                raise ValueError(f"unknown layer {word!r}")
            c = out or c
            side = math.ceil(side / s)
        if side != 1:
            raise ValueError("the arch must end with gap or fl")
        layers.append(nn.Linear(c, 10))
        self.layers = nn.Sequential(*layers)
        if cfg.standardize:
            self.register_buffer("mean", mean.view(1, 3, 1, 1))
            self.register_buffer("std", std.view(1, 3, 1, 1))
        self.standardize = cfg.standardize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.standardize:
            x = (x - self.mean) / self.std
        return self.layers(x)


# The 8 symmetries of a square (the group D4), as (transpose?, flip dims). Views are
# taken in this order: 2 views = the image and its mirror image, 4 adds the vertical
# flips, 8 adds the transposes (the 90-degree turns).
SYMMETRIES = [
    (False, ()),
    (False, (3,)),
    (False, (2,)),
    (False, (2, 3)),
    (True, ()),
    (True, (3,)),
    (True, (2,)),
    (True, (2, 3)),
]


class Views(nn.Module):
    """Answer with the average of the network's scores over several turned copies."""

    def __init__(self, core: nn.Module, views: int) -> None:
        super().__init__()
        self.core = core
        self.views = views

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = 0
        for transpose, dims in SYMMETRIES[: self.views]:
            view = x.transpose(2, 3) if transpose else x
            total = total + self.core(view.flip(dims) if dims else view)
        return total / self.views


# ------------------------------------------------------------- training


def augment(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Random symmetries (and shifts) of a batch of float images, each image its own."""
    n = len(x)

    def coin() -> torch.Tensor:
        return torch.rand(n, 1, 1, 1, device=x.device) < 0.5

    parts = kind.split("+")
    if parts[0] in ("hflip", "flips", "d4"):
        x = torch.where(coin(), x.flip(3), x)
    if parts[0] in ("flips", "d4"):
        x = torch.where(coin(), x.flip(2), x)
    if parts[0] == "d4":
        x = torch.where(coin(), x.transpose(2, 3), x)
    for part in parts[1:]:
        if part.startswith("crop"):  # shift by up to P pixels, filling in by reflection
            pad = int(part[4:])
            side = x.shape[-1]
            padded = F.pad(x, (pad,) * 4, mode="reflect")
            dy, dx = (torch.randint(0, 2 * pad + 1, (2, n, 1), device=x.device)).unbind(0)
            rows = (dy + torch.arange(side, device=x.device))[:, None, :, None]
            cols = (dx + torch.arange(side, device=x.device))[:, None, None, :]
            batch = torch.arange(n, device=x.device)[:, None, None, None]
            chans = torch.arange(3, device=x.device)[None, :, None, None]
            x = padded[batch, chans, rows, cols]
    return x.contiguous(memory_format=torch.channels_last)


def make_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.opt == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.opt == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.opt == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.wd)
    raise ValueError(cfg.opt)


def lr_factor(cfg: Config, step: int, total: int) -> float:
    if cfg.sched == "const":
        return 1.0
    warm = round(cfg.warmup * total)
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def build(cfg: Config, images: torch.Tensor) -> Net:
    mean = std = None
    if cfg.standardize:
        sample = images[:2000].float() / 255
        mean, std = sample.mean((0, 2, 3)), sample.std((0, 2, 3))
    return Net(cfg, mean, std)


def fit(cfg: Config, train_set, seed: int, device: str = "cpu", valid=None):
    """Train a network; return it and its curve (one point per epoch)."""
    torch.manual_seed(seed)
    images, labels = train_set.images.to(device), train_set.labels.to(device)
    model = build(cfg, train_set.images).to(device).to(memory_format=torch.channels_last)
    optimizer = make_optimizer(cfg, model)
    steps = math.ceil(len(images) / cfg.batch)
    total = steps * cfg.epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_factor(cfg, i, total))
    curve = []
    for _ in range(cfg.epochs):
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        for idx in torch.randperm(len(images), device=device).split(cfg.batch):
            if len(idx) < 2:  # batch norm needs two images
                continue
            x = augment(images[idx].float().div_(255), cfg.aug)
            y = labels[idx]
            scores = model(x)
            loss = F.cross_entropy(scores, y, label_smoothing=cfg.smooth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            loss_sum += loss.detach() * len(idx)
            correct += (scores.argmax(1) == y).sum()
        point = {
            "train_loss": round(loss_sum.item() / len(images), 4),
            "train_acc": round(correct.item() / len(images), 4),
        }
        if valid is not None:
            point["valid_acc"] = accuracy(model, valid, device)
        curve.append(point)
    return model, curve


def accuracy(model: nn.Module, dataset, device: str = "cpu", views: int = 1) -> float:
    model.eval()
    net = Views(model, views) if views > 1 else model
    right = 0
    with torch.inference_mode():
        for x, y in zip(dataset.images.split(1000), dataset.labels.split(1000), strict=True):
            x = x.to(device).float().div_(255).contiguous(memory_format=torch.channels_last)
            right += (net(x).argmax(1).cpu() == y).sum().item()
    model.train()
    return round(right / len(dataset), 4)


class Counted:
    """The repo's FLOP counter. On an older PyTorch (the pod has 2.8) its check for
    uncounted ops needs a tag that does not exist yet, so there we count with the same
    counter and formulas alone: the total is the same, only that check is missing."""

    def __enter__(self):
        if hasattr(torch.Tag, "reduction"):
            self.context = counting()
            self.count = self.context.__enter__()
        else:
            self.context = FlopCounterMode(display=False, custom_mapping=FORMULAS)
            self.context.__enter__()
            self.count = None
        return self

    def __exit__(self, *exc):
        self.context.__exit__(*exc)
        self.flops = self.count.flops if self.count else self.context.get_total_flops()


def count_flops(cfg: Config, n_train: int) -> dict[str, int]:
    """FLOPs to train (all epochs) and to answer one image, with the repo's counter."""
    fake = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8)
    model = build(cfg, fake)
    with Counted() as step:
        F.cross_entropy(model(fake.float() / 255), torch.zeros(8, dtype=torch.long)).backward()
    model.eval()
    with Counted() as answer, torch.inference_mode():
        Views(model, cfg.tta)(fake.float() / 255)
    return {
        "train_flops": step.flops // 8 * n_train * cfg.epochs,
        "predict_flops": answer.flops // 8,
        "params": sum(p.numel() for p in model.parameters()),
    }


# ------------------------------------------------------------- the runner

_FULL = None


def full_set():
    global _FULL
    if _FULL is None:
        _FULL = data.train_set()
    return _FULL


def key(cfg: Config, seed: int, split: int) -> str:
    text = json.dumps([asdict(cfg), seed, split], sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def run_one(exp: str, label: str, cfg: Config, seed: int, split: int, device: str) -> dict:
    torch.set_num_threads(2)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    train_set, valid = data.split_for_check(full_set(), seed=split)
    flops = count_flops(cfg, len(train_set))  # first, so a failure here wastes no training
    start = time.time()
    model, curve = fit(cfg, train_set, seed, device, valid)
    seconds = time.time() - start
    return {
        "exp": exp,
        "label": label,
        "seed": seed,
        "split": split,
        "key": key(cfg, seed, split),
        "acc": accuracy(model, valid, device, cfg.tta),
        "acc_1view": curve[-1]["valid_acc"],
        "acc_views": {v: accuracy(model, valid, device, v) for v in (1, 2, 4, 8)},
        "train_acc": curve[-1]["train_acc"],
        "seconds": round(seconds, 1),
        **flops,
        "cfg": asdict(cfg),
        "curve": curve,
    }


def load(*exps: str) -> list[dict]:
    """Every finished run of these experiments (all of them if none named)."""
    paths = [RUNS / f"{e}.jsonl" for e in exps] if exps else sorted(RUNS.glob("*.jsonl"))
    rows = []
    for path in paths:
        if path.exists():
            rows += [json.loads(line) for line in path.read_text().splitlines() if line]
    return rows


def main() -> None:
    import experiments

    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="experiments from experiments.py")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    jobs = []
    for name in args.names:
        done = {row["key"] for row in load(name)}
        for label, cfg, seed, split in experiments.EXPERIMENTS[name]():
            if key(cfg, seed, split) not in done:
                jobs.append((name, label, cfg, seed, split))
    print(f"{len(jobs)} runs to do", flush=True)
    context = get_context("spawn")
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        futures = [pool.submit(run_one, *job, args.device) for job in jobs]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:  # one broken setting must not stop the others
                print(f"FAILED: {error!r}", flush=True)
                continue
            with (RUNS / f"{row['exp']}.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(
                f"{row['exp']:>14} {row['label']:<28} seed {row['seed']} "
                f"acc {row['acc']:.4f} train {row['train_acc']:.4f} {row['seconds']}s",
                flush=True,
            )


def with_(cfg: Config, **changes) -> Config:
    return replace(cfg, **changes)


if __name__ == "__main__":
    main()
