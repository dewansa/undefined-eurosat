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


RECIPE = with_(BN_COS, epochs=30, aug="d4+crop4")  # everything so far


def e6_where():
    """The plain CNN spends 9.4 MFLOP in each stage (half the size, twice the channels).
    Which of those FLOPs earn their keep?"""
    return runs(
        {
            "reference (32 MF)": with_(RECIPE, arch="c16 p c32 p c64 p c128 gap"),
            "stem stride 2 (29 MF)": with_(RECIPE, arch="c16s2 c32 p c64 p c128 gap"),
            "one level smaller (15 MF)": with_(RECIPE, arch="c16s2 p c32 p c64 c128 gap"),
            "half the width (8.9 MF)": with_(RECIPE, arch="c8 p c16 p c32 p c64 gap"),
        }
    )


def e7_blocks():
    """At about 10 MFLOP per image, which kind of block buys the most accuracy?"""
    return runs(
        {
            "plain (8.9 MF)": with_(RECIPE, arch="c8 p c16 p c32 p c64 gap"),
            "plain, smaller images (8.6 MF)": with_(RECIPE, arch="c12s2 p c24 p c48 c96 gap"),
            "residual (11.5 MF)": with_(RECIPE, arch="c8s2 r16s2 r32s2 r64 gap"),
            "separable (11.9 MF)": with_(RECIPE, arch="c16s2 d32 d32 p d64 d64 p d128 d128 gap"),
            "inverted residual (10.0 MF)": with_(
                RECIPE, arch="c16s2 m16e2 p m32e2 m32e2 p m64e2 m64e2 gap"
            ),
        }
    )


PROBE = with_(RECIPE, arch="c8s2 r16s2 r32s2 r64 gap")  # 11.5 MF: cheap, and it underfits


def e8_recipe():
    """Every network so far underfits (training accuracy = check accuracy), so what
    makes each FLOP of training go further? Batch size does not change the FLOPs of an
    epoch, only how many updates they buy. Asked of a cheap probe (95.18 in exp. 7)."""
    sgd = with_(PROBE, opt="sgd", wd=5e-4)
    return runs(
        {
            "batch 32": with_(PROBE, batch=32),
            "batch 128": with_(PROBE, batch=128),
            "adam lr 1e-3": with_(PROBE, lr=1e-3),
            "adam lr 1e-2": with_(PROBE, lr=1e-2),
            "adamw lr 3e-3 wd 0.05": with_(PROBE, opt="adamw", wd=0.05),
            "sgd lr 0.05": with_(sgd, lr=0.05),
            "sgd lr 0.1": with_(sgd, lr=0.1),
            "sgd lr 0.2": with_(sgd, lr=0.2),
            "label smoothing 0.1": with_(PROBE, smooth=0.1),
        }
    )


def e9_residual():
    """Residual blocks beat plain ones at 10 MF (exp. 7), and the 32x32 stage earns its
    FLOPs (exp. 6). Residual networks that keep that stage:"""
    return runs(
        {
            "r at 32, 16, 8 (22.5 MF)": with_(RECIPE, arch="c8s2 r16 r32s2 r64s2 gap"),
            "c at 32, r at 16, 8 (20.3 MF)": with_(RECIPE, arch="c16s2 c16 r32s2 r64s2 gap"),
            "plain, same stages (29 MF)": with_(RECIPE, arch="c16s2 c32 p c64 p c128 gap"),
        }
    )


SGD = with_(RECIPE, opt="sgd", lr=0.1, wd=5e-4)  # exp. 8's choice
PROBE_SGD = with_(SGD, arch=PROBE.arch)  # 11.5 MF; 96.30 at 30 epochs (exp. 8)


def e10_budget():
    """With a fixed training budget of ~345 MFLOP-epochs (the probe's 11.5 MF x 30;
    BuzinD spends ~420), where should the compute go: width, depth, or epochs?"""

    def at(arch: str, mflop: float):
        return with_(SGD, arch=arch, epochs=round(345 / mflop))

    return runs(
        {
            "probe, 20 epochs (230)": with_(PROBE_SGD, epochs=20),
            "probe, 45 epochs (518)": with_(PROBE_SGD, epochs=45),
            "narrow x0.75, 53 ep": at("c6s2 r12s2 r24s2 r48 gap", 6.5),
            "last stage 48, 40 ep": at("c8s2 r16s2 r32s2 r48 gap", 8.7),
            "wide x1.5, 14 ep": at("c12s2 r24s2 r48s2 r96 gap", 25.4),
            "deeper at 8x8, 17 ep": at("c8s2 r16s2 r32s2 r64 r64 gap", 20.9),
            "deeper at 16x16, 18 ep": at("c8s2 r16s2 r32 r32s2 r64 gap", 19.4),
            "r at 32, 16, 8, 15 ep": at("c8s2 r16 r32s2 r64s2 gap", 22.5),
            "plain 32, 16, 8, 12 ep": at("c16s2 c32 p c64 p c128 gap", 29.2),
        }
    )


CANDIDATE = with_(SGD, arch="c8s2 r16s2 r32s2 r48 gap", epochs=40)  # exp. 10's winner


def e11_epochs():
    """How does the candidate (8.7 MF) trade compute for accuracy?"""
    return runs({f"{e} epochs": with_(CANDIDATE, epochs=e) for e in (30, 50, 60)})


def e12_neighbours():
    """Is the candidate's shape a local best? Its neighbours, at the same 40 epochs."""
    return runs(
        {
            "last stage 40 (7.6 MF)": with_(CANDIDATE, arch="c8s2 r16s2 r32s2 r40 gap"),
            "last stage 32 (6.5 MF)": with_(CANDIDATE, arch="c8s2 r16s2 r32s2 r32 gap"),
            "last stage 56 (10.0 MF)": with_(CANDIDATE, arch="c8s2 r16s2 r32s2 r56 gap"),
            "middle 24 (7.6 MF)": with_(CANDIDATE, arch="c8s2 r16s2 r24s2 r48 gap"),
            "first 12, middle 24 (6.8 MF)": with_(CANDIDATE, arch="c8s2 r12s2 r24s2 r48 gap"),
            "stem 12 (9.3 MF)": with_(CANDIDATE, arch="c12s2 r16s2 r32s2 r48 gap"),
        }
    )


# The choice: exp. 12's cheapest tie with the candidate, answering with 4 views
# (8 views added nothing for it on split 11).
CHOSEN = with_(CANDIDATE, arch="c8s2 r16s2 r32s2 r32 gap", tta=4)


def e13_confirm():
    """Does the choice hold on images that played no part in making it (splits 7 and 3)?
    BuzinD's submission scores 97.06 and 96.81 there. The runner-up, for comparison."""
    return runs(
        {"chosen (6.5 MF, 40 ep)": CHOSEN, "runner-up, last stage 48": with_(CANDIDATE, tta=4)},
        splits=(7, 3),
    )


def e14_ablation():
    """Did every part of the choice earn its place? Each removed in turn (split 11)."""
    return runs(
        {
            "no shifts (symmetries only)": with_(CHOSEN, aug="d4"),
            "no augmentation": with_(CHOSEN, aug="none"),
            "Adam 3e-3 instead of SGD": with_(CHOSEN, opt="adam", lr=3e-3, wd=0.0),
            "constant rate 0.1 (no schedule)": with_(CHOSEN, sched="const"),
            "no residual shortcuts": with_(CHOSEN, arch="c8s2 c16s2 c16 c32s2 c32 c32 c32 gap"),
            "no batch norm (standardized input)": with_(CHOSEN, norm=False, standardize=True),
        }
    )


def e15_batch():
    """Training time: on a CPU the early layers (many pixels, few channels) are slow per
    FLOP, and a bigger batch does the same FLOPs in fewer, fuller steps (about 17% faster
    per image at 128). Exp. 8 found batch 128 worse at a fixed rate; with the rate scaled
    with the batch (the linear scaling rule), does it keep the accuracy?"""
    return runs(
        {
            "batch 64, lr 0.1 (the choice)": CHOSEN,
            "batch 128, lr 0.1": with_(CHOSEN, batch=128),
            "batch 128, lr 0.2": with_(CHOSEN, batch=128, lr=0.2),
            "batch 256, lr 0.2": with_(CHOSEN, batch=256, lr=0.2),
            "batch 256, lr 0.4": with_(CHOSEN, batch=256, lr=0.4),
        }
    )


EXPERIMENTS = {
    "0-naive": e0_naive,
    "1-lr": e1_lr,
    "2-inputs": e2_inputs,
    "3-head": e3_head,
    "4-schedule": e4_schedule,
    "5-augment": e5_augment,
    "6-where": e6_where,
    "7-blocks": e7_blocks,
    "8-recipe": e8_recipe,
    "9-residual": e9_residual,
    "10-budget": e10_budget,
    "11-epochs": e11_epochs,
    "12-neighbours": e12_neighbours,
    "13-confirm": e13_confirm,
    "14-ablation": e14_ablation,
    "15-batch": e15_batch,
}
