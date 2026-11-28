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


PLANS = {
    "0-leader": leader,
    "1-size-and-time": size_and_time,
    "2-recipe": recipe,
    "3-shape": shape,
    "4-smaller-longer": smaller_longer,
}
