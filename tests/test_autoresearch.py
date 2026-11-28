"""autoresearch's model.py is the lean copy of the setting its experiments chose.

search.py trained every run of the climb; model.py is what GitHub trains. At the chosen
setting (experiments.FINAL) the two must be the same model: same weights after the same
training, same answers, same compute. Otherwise the experiments would not describe the
submission.
"""

import importlib
import sys
from pathlib import Path

import torch

from eurosat.data import EuroSAT

FOLDER = Path("submissions/autoresearch").resolve()
sys.path.insert(0, str(FOLDER))
model = importlib.import_module("model")
search = importlib.import_module("search")
experiments = importlib.import_module("experiments")


def fake_images(n: int) -> EuroSAT:
    generator = torch.Generator().manual_seed(1)
    images = torch.randint(0, 256, (n, 3, 64, 64), dtype=torch.uint8, generator=generator)
    return EuroSAT(images, torch.randint(0, 10, (n,), generator=generator))


def test_the_chosen_setting_is_the_submission() -> None:
    m, c = model.Config(), experiments.FINAL
    stem, *blocks = zip(m.widths, m.strides, strict=True)
    arch = f"c{stem[0]}s{stem[1]} " + " ".join(
        f"r{w}" + (f"s{s}" if s > 1 else "") for w, s in blocks
    )
    assert (arch + " gap", m.epochs, m.batch_size, m.lr, m.weight_decay, m.warmup, m.views) == (
        c.arch,
        c.epochs,
        c.batch,
        c.lr,
        c.wd,
        c.warmup,
        c.tta,
    )
    assert (c.opt, c.sched, c.norm, c.aug, c.smooth) == ("sgd", "cos", True, f"d4+crop{m.shift}", 0)


def test_both_train_to_the_same_weights_and_answers() -> None:
    images = fake_images(300)
    short = search.with_(experiments.FINAL, epochs=2)
    ours = model.train(images, lambda **_: None, model.Config(epochs=2))
    theirs, _ = search.fit(short, images, seed=model.Config().seed)
    for a, b in zip(ours.network.state_dict().values(), theirs.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    x = images.images[:20].float() / 255
    with torch.inference_mode():
        assert torch.equal(ours(x), search.Views(theirs.eval(), short.tta)(x))


def test_both_cost_the_same_compute() -> None:
    counted = search.count_flops(experiments.FINAL, n_train=21600)
    with search.Counted() as answer, torch.inference_mode():
        model.Views(model.make_network(model.Config()).eval(), model.Config().views)(
            torch.rand(8, 3, 64, 64)
        )
    assert answer.flops // 8 == counted["predict_flops"]
