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
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in load(*exps):
        if split is None or row["split"] == split:
            groups[(row["exp"], row["label"])].append(row)
    table = []
    for (exp, label), rows in groups.items():
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
