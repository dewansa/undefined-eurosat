"""The lab bench for demo 2: run an experiment, count its compute, keep the result.

GitHub never runs this file; it only runs model.py. The notebooks use it like this:

    import lab
    lab.run(lab.leader(epochs=8), seed=0)      # one training run: a dict of results
    lab.load("3-width")                        # every saved run of one experiment

From a terminal, run a whole experiment (see plans.py) on a GPU, 6 runs at a time:

    python lab.py 3-width --device cuda --workers 6

Each finished run is one line in runs/<experiment>.jsonl. Runs already in the file
are skipped, so an interrupted experiment continues where it stopped.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model  # noqa: E402
from model import Config  # noqa: E402

from eurosat import data  # noqa: E402

# The leader's settings (BuzinD's model.py), where every experiment starts. model.py's
# defaults are our final choice and will change; these stay fixed, so that each saved run
# keeps its meaning: a run's "settings" are what it changed from the leader.
LEADER = {
    "stem": "conv",
    "stem_stride": 2,
    "stem_pool": True,
    "block": "basic",
    "widths": (24, 48, 96),
    "depths": (1, 1, 1),
    "expand": 4,
    "zero_init": False,
    "optimizer": "adamw",
    "epochs": 28,
    "batch_size": 128,
    "lr": 0.006,
    "weight_decay": 0.01,
    "pct_start": 0.2,
    "label_smoothing": 0.0,
    "mixup": 0.0,
    "ema": 0.0,
    "augment": True,
    "low_res": 0.0,
    "tta": 8,
    "valid_images": 0,
}


def leader(**changes) -> Config:
    """The leader's settings, with some changed: leader(epochs=8)."""
    return Config(**{**LEADER, **changes})


RUNS = Path(os.environ.get("LAB_RUNS", HERE / "runs"))  # where results are saved
CHOOSE_SPLIT = 1  # we choose settings on this split of the training images...
CONFIRM_SPLITS = (7, 3)  # ...and confirm the final choice on these (7 is `uv run score`'s)


class NoLog:
    """Stands in for the leaderboard's log(): keeps only the last epoch's numbers.

    The last epoch's training accuracy, next to the check accuracy, says whether a model
    is memorizing (training accuracy far above check accuracy) or still learning.
    """

    def __init__(self) -> None:
        self.last = {}

    def validate_on(self, images, labels) -> None:
        pass

    def __call__(self, model=None, /, **metrics) -> None:
        self.last = metrics


# ---------------------------------------------------------------- counting compute


def flops_of(fn) -> int:
    """Run fn and count its FLOPs with PyTorch's counter, the one the scorer builds on.

    For convolutions and matrix multiplies (all our networks do) its count is the
    scorer's count; notebook 1 checks this against the leaderboard.
    """
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def step_flops(cfg: Config, size: int, batch: int = 4) -> float:
    """FLOPs of one learning step, per image, for images of size x size pixels.

    A convolution's work grows in proportion to the number of images, so 4 images
    tell us the cost of any batch. Forward and backward pass, like the scorer counts.
    """
    net = model.make_model(cfg).train()
    x = torch.rand(batch, 3, size, size)
    y = torch.zeros(batch, dtype=torch.long)

    def step():
        torch.nn.functional.cross_entropy(net(x), y).backward()

    return flops_of(step) / batch


def train_flops(cfg: Config, n_train: int) -> int:
    """The compute train() does, from the cost of one step: no need to train."""
    n = n_train - cfg.valid_images
    small = round(cfg.low_res * cfg.epochs)
    big = cfg.epochs - small
    return round(n * (small * step_flops(cfg, 32) + big * step_flops(cfg, 64)))


def predict_flops(cfg: Config) -> int:
    """The compute to label one test image, with every orientation the model averages."""
    net = model.Average(model.make_model(cfg), cfg.tta).eval()
    with torch.inference_mode():
        return flops_of(lambda: net(torch.rand(1, 3, 64, 64)))


def parameters(cfg: Config) -> int:
    return sum(p.numel() for p in model.make_model(cfg).parameters())


# ---------------------------------------------------------------- running


_FULL = None


def full_set() -> data.EuroSAT:
    global _FULL
    if _FULL is None:
        _FULL = data.train_set()
    return _FULL


def accuracy(
    net: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    """Loss and accuracy on images fed like the test images: floats from 0 to 1."""
    net.eval()
    loss, correct = 0.0, 0
    with torch.inference_mode():
        for x, y in zip(images.split(500), labels.split(500), strict=True):
            out = net(x.float() / 255)
            loss += torch.nn.functional.cross_entropy(out, y, reduction="sum").item()
            correct += (out.argmax(1) == y).sum().item()
    return loss / len(labels), correct / len(labels)


def changes(cfg: Config) -> dict:
    """The settings that differ from the leader's: what makes this run special."""
    out = {}
    for name, start in LEADER.items():
        value = getattr(cfg, name)
        if value != start:
            out[name] = list(value) if isinstance(value, tuple) else value
    return out


def run(cfg: Config, seed: int = 0, split: int = CHOOSE_SPLIT, quick: bool = False) -> dict:
    """Train once on one split, check on its held-back images, and count the compute.

    quick=True trains on a quarter of the images, for a fast first look on a laptop.
    """
    train_set, check_set = data.split_for_check(full_set(), seed=split)
    if quick:
        train_set = train_set.subset(torch.arange(len(train_set) // 4))
    torch.manual_seed(seed)
    start = time.perf_counter()
    log = NoLog()
    net = model.train(train_set, log, cfg)
    if cfg.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    # The same trained weights, answering with 1, 2, 4 or 8 orientations: a fair
    # comparison of what each extra orientation buys, with no luck of training in it.
    images, labels = check_set.images.to(cfg.device), check_set.labels.to(cfg.device)
    by_views = {}
    for views in (1, 2, 4, 8):
        loss, acc = accuracy(model.Average(net.core, views).to(cfg.device), images, labels)
        by_views[views] = round(acc, 4)
        if views == cfg.tta:
            chosen = acc, loss
    acc, loss = chosen
    return {
        "settings": changes(cfg),
        "seed": seed,
        "split": split,
        "quick": quick,
        "accuracy": round(acc, 4),
        "loss": round(loss, 4),
        "accuracy_by_views": by_views,
        "train_accuracy": round(log.last.get("train_accuracy", float("nan")), 4),
        "train_loss": round(log.last.get("train_loss", float("nan")), 4),
        "train_flops": train_flops(cfg, len(train_set)),
        "train_flops_full": train_flops(cfg, 21600),  # what GitHub will count: all images
        "predict_flops": predict_flops(cfg),
        "parameters": parameters(cfg),
        "seconds": round(seconds, 1),
        "device": cfg.device,
    }


def load(experiment: str):
    """Every saved run of one experiment, as a table (pandas DataFrame)."""
    import pandas as pd

    rows = [json.loads(line) for line in (RUNS / f"{experiment}.jsonl").read_text().splitlines()]
    table = pd.DataFrame(rows)
    table["label"] = [r.get("label", "") for r in rows]
    return table


def setting(table, name: str) -> list:
    """One setting of every run in a table, including runs that left it as the leader had it."""
    return [s.get(name, LEADER[name]) for s in table["settings"]]


def summary(table, by: str = "label"):
    """Mean accuracy per setting, its standard error, and its compute.

    The standard error (se) says how far the mean of these few runs may be from the mean
    of infinitely many: about 2 out of 3 times within 1 se, 19 out of 20 within 2 se.
    Training compute is for all 21,600 images, as GitHub counts it.
    """
    out = table.groupby(by, sort=False).agg(
        accuracy=("accuracy", "mean"),
        sd=("accuracy", "std"),
        runs=("accuracy", "size"),
        train_tflop=("train_flops_full", lambda f: f.mean() / 1e12),
        predict_mflop=("predict_flops", lambda f: f.mean() / 1e6),
        parameters=("parameters", "mean"),
    )
    out["se"] = out["sd"] / out["runs"] ** 0.5
    out[["accuracy", "sd", "se"]] *= 100
    out = out[["accuracy", "se", "sd", "runs", "train_tflop", "predict_mflop", "parameters"]]
    return out.round({"accuracy": 2, "se": 2, "sd": 2, "train_tflop": 2, "predict_mflop": 1})


def compare(table, a: str, b: str, by: str = "label") -> str:
    """Is setting b really better than setting a, or is the difference luck?

    The difference of two means has a standard error of sqrt(se_a² + se_b²). A difference
    of more than 2 of those happens by luck about 1 time in 20: we call it real.
    """
    s = summary(table, by)
    diff = s.loc[b, "accuracy"] - s.loc[a, "accuracy"]
    se = (s.loc[a, "se"] ** 2 + s.loc[b, "se"] ** 2) ** 0.5
    verdict = "real" if abs(diff) > 2 * se else "could be luck"
    return f"{b} vs {a}: {diff:+.2f} points ± {se:.2f} -> {verdict}"


# ---------------------------------------------------------------- experiments from a terminal


def key(job: dict) -> tuple:
    return (job["label"], job["seed"], job["split"])


def work(job: dict) -> dict:
    torch.set_num_threads(job.pop("threads"))
    # Plain float32 on the GPU too (no TF32 shortcuts), so results carry over to the CPU.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    cfg = leader(**job["config"], device=job.pop("device"))
    result = run(cfg, seed=job["seed"], split=job["split"])
    return {"experiment": job["experiment"], "label": job["label"], **result}


def main() -> None:
    import multiprocessing as mp

    from plans import PLANS

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiments", nargs="+", help="names in plans.py")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1, help="runs at the same time")
    parser.add_argument("--dry", action="store_true", help="list the runs, count their compute")
    args = parser.parse_args()

    full_set()  # download once, before the workers start
    RUNS.mkdir(exist_ok=True)
    jobs = []
    for name in args.experiments:
        path = RUNS / f"{name}.jsonl"
        done = set()
        if path.exists():
            done = {key(json.loads(line)) for line in path.read_text().splitlines()}
        for job in PLANS[name]():
            job["experiment"] = name
            if key(job) not in done:
                jobs.append(job)
    threads = max(1, (os.cpu_count() or 1) // args.workers)
    for job in jobs:
        job.update(device=args.device, threads=threads)
    total = sum(train_flops(leader(**j["config"]), 17280) for j in jobs)
    print(f"{len(jobs)} runs to do, {total / 1e12:.0f} TFLOP of training", flush=True)
    if args.dry:
        for job in jobs:
            print(job["experiment"], job["label"], job["seed"], job["split"], job["config"])
        return

    start = time.perf_counter()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool:
        for i, result in enumerate(pool.imap_unordered(work, jobs), 1):
            with (RUNS / f"{result['experiment']}.jsonl").open("a") as file:
                file.write(json.dumps(result) + "\n")
            minutes = (time.perf_counter() - start) / 60
            print(
                f"[{i}/{len(jobs)} {minutes:.1f} min] {result['experiment']} {result['label']}"
                f" seed {result['seed']}: {100 * result['accuracy']:.2f}%"
                f" ({result['seconds']:.0f} s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
