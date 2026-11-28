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
    return tuple(round(w * m) for w in (24, 48, 96))


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


PLANS = {
    "0-leader": leader,
    "1-size-and-time": size_and_time,
}
