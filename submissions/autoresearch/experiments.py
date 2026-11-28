"""Every experiment of the climb: a name, and the settings it trains (3 seeds each).

Each experiment asks one question; NOTES.md has the question, the prediction, the
result and the decision. Settings are chosen on split 11 (4,320 images held back from
training), never on split 7, which `uv run score` uses, nor on split 3: those two are
kept to confirm the choice on images that played no part in making it.
"""

from __future__ import annotations

from search import Config, with_

CHOOSE = 11  # the split every choice is made on
SEEDS = (0, 1, 2)

NAIVE = Config()  # c16 p c32 p c64 p fl, Adam 1e-3, 10 epochs, batch 64


def runs(settings: dict[str, Config], seeds=SEEDS, splits=(CHOOSE,)):
    return [(label, cfg, s, sp) for label, cfg in settings.items() for s in seeds for sp in splits]


def e0_naive():
    """Where does the naive CNN stand, and is it under- or over-fitting?"""
    return runs({"naive 10 ep": NAIVE, "naive 30 ep": with_(NAIVE, epochs=30)})


def e1_lr():
    """Is the default Adam rate (1e-3) right for the naive CNN?"""
    return runs({f"lr {lr:g}": with_(NAIVE, lr=lr) for lr in (3e-4, 1e-3, 3e-3, 1e-2)})


def e2_inputs():
    """Does it help to standardize the input, or to normalize every layer (batch norm)?"""
    return runs(
        {
            "standardize": with_(NAIVE, standardize=True),
            "batch norm": with_(NAIVE, norm=True),
            "batch norm lr 3e-3": with_(NAIVE, norm=True, lr=3e-3),
            "batch norm lr 1e-2": with_(NAIVE, norm=True, lr=1e-2),
        }
    )


def e3_head():
    """Flatten (65k weights in the head) or global average pool (640)?"""
    bn = with_(NAIVE, norm=True)
    return runs(
        {
            "flatten": bn,
            "gap": with_(bn, arch="c16 p c32 p c64 p gap"),
            "gap, 2x wider": with_(bn, arch="c32 p c64 p c128 p gap"),
            "gap, conv at 8x8": with_(bn, arch="c16 p c32 p c64 p c128 gap"),
        }
    )


def e4_schedule():
    """Batch norm's check accuracy swings 57-87 between epochs at a constant rate.
    Does warming up and then annealing the rate to zero (cosine) settle it?"""
    cos = with_(NAIVE, sched="cos", warmup=0.1)
    bn = with_(cos, norm=True)
    return runs(
        {
            "standardize, cos 1e-3": with_(cos, standardize=True),
            "standardize, cos 3e-3": with_(cos, standardize=True, lr=3e-3),
            "bn, cos 1e-3": bn,
            "bn, cos 3e-3": with_(bn, lr=3e-3),
            "bn, cos 1e-2": with_(bn, lr=1e-2),
            "bn, cos 3e-3, gap": with_(bn, lr=3e-3, arch="c16 p c32 p c64 p gap"),
            "bn, cos 3e-3, gap + conv 8x8": with_(bn, lr=3e-3, arch="c16 p c32 p c64 p c128 gap"),
        }
    )


BN_COS = with_(NAIVE, norm=True, sched="cos", warmup=0.1, lr=3e-3)


def e5_augment():
    """At 30 epochs the naive CNN memorizes (train 94, check 84). A satellite photo has
    no "up": do flipped and turned copies (and small shifts) stop the memorizing?"""
    base = with_(BN_COS, epochs=30)
    return runs(
        {
            f"30 ep, {aug}": with_(base, aug=aug)
            for aug in ("none", "hflip", "flips", "d4", "d4+crop4")
        }
    )


EXPERIMENTS = {
    "0-naive": e0_naive,
    "1-lr": e1_lr,
    "2-inputs": e2_inputs,
    "3-head": e3_head,
    "4-schedule": e4_schedule,
    "5-augment": e5_augment,
}
