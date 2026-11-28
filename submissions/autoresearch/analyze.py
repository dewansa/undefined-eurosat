"""Summaries of finished runs: one line per setting, averaged over its seeds.

python analyze.py 0-naive 1-lr        # print the tables
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict

import torch
from search import Config, build, load
from torch.utils.flop_counter import FlopCounterMode


def mean_se(values: list[float]) -> tuple[float, float]:
    """The mean, and its standard error (how far luck alone moves the mean)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def summary(*exps: str, split: int | None = None) -> list[dict]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in load(*exps):
        if split is None or row["split"] == split:
            groups[(row["exp"], row["label"], row["split"])].append(row)
    table = []
    for (exp, label, on), rows in groups.items():
        if on != 11:  # every choice is made on split 11; name any other split
            label = f"{label}, split {on}"
        acc, se = mean_se([100 * r["acc"] for r in rows])
        table.append(
            {
                "exp": exp,
                "label": label,
                "runs": len(rows),
                "acc": acc,
                "se": se,
                "acc_1view": mean_se([100 * r["acc_1view"] for r in rows])[0],
                "train_acc": mean_se([100 * r["train_acc"] for r in rows])[0],
                "train_tflop": rows[0]["train_flops"] / 1e12,
                "predict_mflop": rows[0]["predict_flops"] / 1e6,
                "params": rows[0]["params"],
                "gpu_s": mean_se([r["seconds"] for r in rows])[0],
                "runs_": rows,
            }
        )
    return table


def layer_flops(cfg: Config) -> list[tuple[str, float]]:
    """Forward FLOPs of one 64x64 image, layer by layer (in MFLOP), named by the arch words."""
    model = build(cfg, torch.zeros(1, 3, 64, 64, dtype=torch.uint8)).eval()
    with FlopCounterMode(display=False) as counter, torch.inference_mode():
        model(torch.rand(1, 3, 64, 64))
    counts = counter.get_flop_counts()
    words = cfg.arch.split()
    out = []
    for i, word in enumerate(words):
        out.append((word, sum(counts.get(f"Net.layers.{i}", {}).values()) / 1e6))
    # the head is the last layer (after gap or fl, which adds a Flatten for gap)
    head = f"Net.layers.{len(model.layers) - 1}"
    out.append(("head", sum(counts.get(head, {}).values()) / 1e6))
    return out


def show(*exps: str, split: int | None = None) -> None:
    print(
        f"{'experiment':<14}{'setting':<34}{'n':>2} {'check acc':>12} {'1 view':>7}"
        f" {'train':>6} {'TFLOP':>6} {'MFLOP':>6} {'params':>8} {'gpu s':>6}"
    )
    for t in summary(*exps, split=split):
        print(
            f"{t['exp']:<14}{t['label']:<34}{t['runs']:>2} {t['acc']:6.2f} ±{t['se']:4.2f}"
            f" {t['acc_1view']:7.2f} {t['train_acc']:6.2f} {t['train_tflop']:6.2f}"
            f" {t['predict_mflop']:6.1f} {t['params']:8d} {t['gpu_s']:6.0f}"
        )


if __name__ == "__main__":
    show(*sys.argv[1:])


def cpu_seconds(cfg: Config, images: int = 21600, steps: int = 25) -> float:
    """Estimated seconds for this machine's CPU to train cfg on `images` images.

    Times `steps` training steps after a warm-up and scales up. GitHub's computer is
    slower: BuzinD's submission takes 184 s here and 407 s there (2.2 times).
    """
    import time

    from search import augment, make_optimizer
    from torch.nn import functional as F

    torch.manual_seed(0)
    x8 = torch.randint(0, 256, (cfg.batch, 3, 64, 64), dtype=torch.uint8)
    y = torch.randint(0, 10, (cfg.batch,))
    model = build(cfg, x8).to(memory_format=torch.channels_last)
    optimizer = make_optimizer(cfg, model)

    def step() -> None:
        x = augment(x8.float().div_(255), cfg.aug)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(5):
        step()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    per_image = (time.perf_counter() - start) / (steps * cfg.batch)
    return per_image * images * cfg.epochs


# ------------------------------------------------------------------ plots
# Colours: ours in blue, the bar (BuzinD) in orange; text and grid stay gray.
OURS, BAR, INK, GRID = "#257de0", "#f47230", "#444444", "#e5e5e5"


def _axes(ax, xlabel: str) -> None:
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, length=0)


def plot_settings(*exps: str, bar: float | None = None, views: int | None = None):
    """One dot per setting: mean check accuracy, with a line of +-1 standard error."""
    import matplotlib.pyplot as plt

    rows = []
    for t in summary(*exps):
        group = t["runs_"]
        values = [100 * (r["acc_views"][str(views)] if views else r["acc"]) for r in group]
        rows.append((t["label"], *mean_se(values)))
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 0.9))
    for i, (_, acc, se) in enumerate(rows):
        ax.errorbar(acc, i, xerr=se if se == se else 0, fmt="o", color=OURS, ms=6, lw=2)
        ax.annotate(
            f"{acc:.2f}",
            (acc, i),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    if bar is not None:
        ax.axvline(bar, color=BAR, lw=2)
        ax.annotate(
            "BuzinD on this split",
            (bar, len(rows) - 0.5),
            xytext=(4, 0),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    _axes(
        ax,
        "check accuracy, % (mean of 3 seeds, ±1 standard error)"
        + (f", {views} views" if views else ""),
    )
    fig.tight_layout()
    return fig


def plot_curves(exp: str, labels: list[str], seed: int = 0):
    """Training accuracy (dashed) and check accuracy (solid), epoch by epoch."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 2.8), sharey=True)
    axes = [axes] if len(labels) == 1 else axes
    for ax, label in zip(axes, labels, strict=True):
        run = next(r for r in load(exp) if r["label"] == label and r["seed"] == seed)
        epochs = range(1, len(run["curve"]) + 1)
        ax.plot(
            epochs,
            [100 * p["train_acc"] for p in run["curve"]],
            "--",
            color=INK,
            lw=1.5,
            label="training (augmented)",
        )
        ax.plot(
            epochs, [100 * p["valid_acc"] for p in run["curve"]], color=OURS, lw=2, label="check"
        )
        ax.set_title(label, fontsize=9, color=INK)
        ax.grid(color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("epoch", color=INK)
    axes[0].set_ylabel("accuracy, %", color=INK)
    axes[0].set_ylim(50, 100)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_frontier(
    exps: list[str], bar_acc: float, bar_tflop: float, views: int = 4, chosen: str = ""
):
    """Accuracy against training compute (on all 21,600 images) for every setting."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for t in summary(*exps):
        group = t["runs_"]
        acc = mean_se([100 * r["acc_views"][str(views)] for r in group])[0]
        tflop = t["train_tflop"] * 21600 / 17280
        ax.plot(tflop, acc, "o", color=OURS, ms=5, alpha=0.75)
        if t["label"] == chosen:
            ax.plot(tflop, acc, "o", color=OURS, ms=10, mfc="none", mew=2)
            ax.annotate(
                "the choice",
                (tflop, acc),
                xytext=(-8, 8),
                textcoords="offset points",
                ha="right",
                color=INK,
                fontsize=9,
            )
    ax.plot(bar_tflop, bar_acc, "D", color=BAR, ms=8)
    ax.annotate(
        "BuzinD (8 views)",
        (bar_tflop, bar_acc),
        xytext=(6, -3),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    ax.set_xscale("log")
    ticks = [t for t in (5, 10, 15, 20, 30, 40, 60, 80) if t < 1.3 * ax.get_xlim()[1]]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel(f"check accuracy, % ({views} views)", color=INK)
    _axes(ax, "training compute on all 21,600 images, TFLOP (log scale)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    return fig
