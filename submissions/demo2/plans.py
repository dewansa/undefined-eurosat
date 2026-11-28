"""Every experiment of demo 2, as a list of training runs.

Each experiment asks one question. Its runs differ in one thing (or in a grid of two),
and each setting runs with 3 seeds, so we can tell a real difference from luck.
Run one with `python lab.py <name> --device cuda --workers 6`; results go to runs/.

The order is the order we asked the questions in: each one was written after
looking at the answers to the ones before it.
"""

SEEDS = (0, 1, 2)


def runs(label: str, seeds=SEEDS, split: int = 1, **config) -> list[dict]:
    return [{"label": label, "config": config, "seed": s, "split": split} for s in seeds]


def scaled(m: float) -> tuple[int, ...]:
    """The leader's widths (24, 48, 96), times m."""
    return tuple(round(w * m) for w in (24, 48, 96))  # 0.33 -> (8, 16, 32)


def leader():
    """Start where the leaderboard is: the top submission's settings, retrained 3 times."""
    return runs("leader")


def size_and_time():
    """Bigger network, or more epochs? The same network family, smaller and larger,
    each trained for 4, 8 and 16 epochs. (28 epochs at width 1 is the leader.)"""
    jobs = []
    for m in (0.5, 0.75, 1.0, 1.5):
        for epochs in (4, 8, 16):
            jobs += runs(f"width {m} x {epochs} epochs", widths=scaled(m), epochs=epochs)
    return jobs


# The probe: half the leader's width for 16 epochs, about 4 TFLOP. Inside the budget,
# and quick, so we can try many changes on it. One change per setting.
PROBE = {"widths": (12, 24, 48), "epochs": 16}


def recipe():
    """How we train, at a fixed cost: each setting changes one thing from the probe.
    None of these change the FLOPs, except low_res (which saves some)."""
    changes = {
        "probe": {},
        "lr 0.003": {"lr": 0.003},
        "lr 0.012": {"lr": 0.012},
        "lr 0.024": {"lr": 0.024},
        "batch 64": {"batch_size": 64},
        "batch 256": {"batch_size": 256},
        "no weight decay": {"weight_decay": 0.0},
        "weight decay 0.05": {"weight_decay": 0.05},
        "label smoothing 0.1": {"label_smoothing": 0.1},
        "average weights 0.99": {"ema": 0.99},
        "average weights 0.998": {"ema": 0.998},
        "blocks start as identity": {"zero_init": True},
        "no turning or mirroring": {"augment": False},
        "half the epochs at 32x32": {"low_res": 0.5},
    }
    return [job for label, c in changes.items() for job in runs(label, **{**PROBE, **c})]


def shape():
    """Where to put the compute: other shapes of plain convolutional networks, 16 epochs.
    (Depthwise convolutions are out: the FLOP counter charges their backward pass as if
    they were full convolutions, about 30 times what they compute.)"""
    shapes = {
        "probe": {},
        "first layer: 4x4 squares": {"stem": "patch", "stem_stride": 4},
        "no stage 1": {"depths": (0, 1, 1)},
        "no stage 1, wider (16, 32, 64)": {"depths": (0, 1, 1), "widths": (16, 32, 64)},
        "wider late (12, 32, 64)": {"widths": (12, 32, 64)},
        "deeper (1, 2, 2)": {"depths": (1, 2, 2)},
    }
    return [job for label, c in shapes.items() for job in runs(label, **{**PROBE, **c})]


def smaller_longer():
    """Experiment 1 said: at the same compute, more epochs beat more width. How far does
    that go? Narrower networks still, trained for longer."""
    jobs = []
    for m, epochs in ((0.25, (32, 64, 128)), (0.33, (20, 40, 80)), (0.5, (32,))):
        for e in epochs:
            jobs += runs(f"width {m} x {e} epochs", widths=scaled(m), epochs=e)
    return jobs


def batch_and_lr():
    """Experiment 2: batch 64 beat batch 128 by a point, at the same FLOPs. Smaller
    batches make more (noisier) learning steps per epoch. Does it go further, and does a
    smaller batch want another learning rate? A grid of the two, at the probe."""
    jobs = []
    for batch in (32, 64, 128):
        for lr in (0.003, 0.006, 0.012, 0.024):
            jobs += runs(f"batch {batch}, lr {lr}", **PROBE, batch_size=batch, lr=lr)
    return jobs


def low_res_fair():
    """Experiment 2: training the first half of the epochs on 32x32 images lost a point,
    but it also saved 37% of the compute. A fair test spends that saving on more epochs:
    every setting here costs about 4 TFLOP, like the probe."""
    return (
        runs("probe: 16 epochs at 64x64", **PROBE)
        + runs("26 epochs, half at 32x32", widths=(12, 24, 48), epochs=26, low_res=0.5)
        + runs("36 epochs, 3/4 at 32x32", widths=(12, 24, 48), epochs=36, low_res=0.75)
    )


# What experiments 2 and 5 chose: batch 64 and a high learning rate (they change no FLOPs).
TRAIN = {"batch_size": 64, "lr": 0.024}


def combined():
    """Put it together. Every setting fills the budget: as many epochs as fit in 5 TFLOP
    (counted on all 21,600 images, as GitHub will). Width 1/3 or 1/2 (experiment 4), and
    the share of epochs on 32x32 images (experiment 6). Plus one step up the learning
    rate, in case the best one lies past the edge of experiment 5's grid."""
    fits = {  # (widths, low_res): epochs that fit in 5 TFLOP, from lab.train_flops
        ((8, 16, 32), 0.5): 68,
        ((8, 16, 32), 0.75): 98,
        ((8, 16, 32), 0.875): 123,
        ((12, 24, 48), 0.5): 32,
        ((12, 24, 48), 0.75): 45,
        ((12, 24, 48), 0.875): 59,
    }
    jobs = []
    for (widths, low), epochs in fits.items():
        label = f"width {widths[0] / 24:.2g}, {epochs} epochs, {low:g} at 32x32"
        jobs += runs(label, **TRAIN, widths=widths, epochs=epochs, low_res=low)
        if low == 0.75:
            config = {**TRAIN, "lr": 0.048, "widths": widths, "epochs": epochs, "low_res": low}
            jobs += runs(label + ", lr 0.048", **config)
    return jobs


# The best of experiment 7 (within luck of the others): width 1/2, 45 epochs, 3/4 of
# them on 32x32 images. About 4.9 TFLOP.
CANDIDATE = {**TRAIN, "widths": (12, 24, 48), "epochs": 45, "low_res": 0.75}


def regularize():
    """Long trainings can memorize. Tricks that fight memorizing cost no FLOPs, and were
    neutral at 16 epochs (experiment 2); at 45 they may not be. 6 seeds each: the
    differences we look for now are a few tenths of a point."""
    changes = {
        "candidate": {},
        "label smoothing 0.1": {"label_smoothing": 0.1},
        "average weights 0.999": {"ema": 0.999},
        "weight decay 0.05": {"weight_decay": 0.05},
        "mixup 0.2": {"mixup": 0.2},
    }
    seeds = range(6)
    return [j for label, c in changes.items() for j in runs(label, seeds, **CANDIDATE, **c)]


def optimizer():
    """Every run so far used AdamW. SGD with momentum is the classic choice for residual
    networks; its learning rates live on another scale, so we try a range."""
    jobs = runs("candidate (AdamW, lr 0.024)", **CANDIDATE)
    for lr in (0.05, 0.1, 0.2, 0.4):
        config = {**CANDIDATE, "optimizer": "sgd", "lr": lr, "weight_decay": 5e-4}
        jobs += runs(f"SGD, lr {lr}", **config)
    return jobs


def budget():
    """What does our recipe give for half, the same, and twice the compute? Together with
    experiments 1 and 4, this draws the recipe's own trade-off between cost and accuracy."""
    return [
        job
        for epochs in (22, 45, 90)
        for job in runs(f"{epochs} epochs", **{**CANDIDATE, "epochs": epochs})
    ]


PLANS = {
    "0-leader": leader,
    "1-size-and-time": size_and_time,
    "2-recipe": recipe,
    "3-shape": shape,
    "4-smaller-longer": smaller_longer,
    "5-batch-and-lr": batch_and_lr,
    "6-low-res-fair": low_res_fair,
    "7-combined": combined,
    "8-regularize": regularize,
    "9-optimizer": optimizer,
    "10-budget": budget,
}
