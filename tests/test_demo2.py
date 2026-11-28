"""demo2's model.py is the lean copy of its bench.py, the version its experiments trained.

At the chosen settings the two must be the same model: same weights after the same
training, same answers, same compute. Otherwise the experiments would not describe the
submission.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from eurosat.data import EuroSAT

DEMO2 = Path("submissions/demo2")


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"demo2_{name}", DEMO2 / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model, bench = load("model"), load("bench")


class Log:
    def validate_on(self, images, labels) -> None:
        pass

    def __call__(self, *model, **metrics) -> None:
        pass


def flops(fn) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def test_the_bench_defaults_are_the_submission() -> None:
    m, b = model.Config(), bench.Config()
    assert {
        "widths": b.widths,
        "epochs": b.epochs,
        "small epochs": round(b.low_res * b.epochs),
        "batch size": b.batch_size,
        "learning rate": b.lr,
        "warmup": b.pct_start,
        "weight decay": b.weight_decay,
    } == {
        "widths": m.widths,
        "epochs": m.epochs,
        "small epochs": m.small_epochs,
        "batch size": m.batch_size,
        "learning rate": m.lr,
        "warmup": m.warmup,
        "weight decay": m.weight_decay,
    }
    # Everything model.py does not have must be switched off (or set as model.py has it).
    fixed = {
        "stem": "conv",
        "stem_stride": 2,
        "stem_pool": True,
        "block": "basic",
        "depths": (1, 1, 1),
        "shortcut": True,
        "zero_init": False,
        "optimizer": "sgd",
        "label_smoothing": 0.0,
        "mixup": 0.0,
        "ema": 0.0,
        "augment": True,
        "tta": 4,
        "valid_images": 0,
    }
    assert {name: getattr(b, name) for name in fixed} == fixed


@pytest.fixture(scope="module")
def both_trained():
    """Both files, trained from the same seed on the same small set: 2 epochs, 1 small."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (200, 3, 64, 64), dtype=torch.uint8, generator=generator)
    labels = torch.randint(0, 10, (200,), generator=generator)
    train_set = EuroSAT(images, labels)
    torch.manual_seed(0)
    lean = model.train(train_set, Log(), model.Config(epochs=2, small_epochs=1))
    torch.manual_seed(0)
    full = bench.train(train_set, Log(), bench.Config(epochs=2, low_res=0.5))
    return lean, full


def test_both_train_to_the_same_weights(both_trained) -> None:
    lean, full = both_trained
    lean_weights = list(lean.network.state_dict().values())
    full_weights = list(full.core.state_dict().values())
    assert len(lean_weights) == len(full_weights)
    for a, b in zip(lean_weights, full_weights, strict=True):
        assert torch.equal(a, b)


def test_both_give_the_same_answers(both_trained) -> None:
    lean, full = both_trained
    x = torch.rand(8, 3, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(lean(x), full(x))


def test_both_cost_the_same_compute() -> None:
    lean, full = model.make_network(model.Config().widths), bench.make_model(bench.Config())

    def step(net, size):
        x, y = torch.rand(4, 3, size, size), torch.zeros(4, dtype=torch.long)
        return lambda: torch.nn.functional.cross_entropy(net(x), y).backward()

    for size in (32, 64):
        assert flops(step(lean, size)) == flops(step(full, size))
    with torch.inference_mode():
        x = torch.rand(1, 3, 64, 64)
        answer = flops(lambda: model.AverageOverTurns(lean.eval())(x))
        assert answer == flops(lambda: bench.Average(full.eval(), 4)(x))
