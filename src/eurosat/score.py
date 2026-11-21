"""Score a submission: train it, then test it on images it has never seen.

On your computer:

    uv run score submissions/<your-github-name>

The real test labels are secret, so this holds back 20% of the training images,
trains on the rest, and checks the model on the held-back images. The score is
close to, but not the same as, the leaderboard's.

    --quick        train on a quarter of the images: a fast first try
    --flops        also count the compute (FLOPs); trains a second time
    --report DIR   save pictures: your first layer's filters and the mistakes

GitHub runs `--official`: train on ALL training images, then save the model's
answers for the test images. A separate job that never runs student code compares
those answers with the secret labels.
"""

import argparse
import importlib.util
import inspect
import json
import math
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

import torch
from torch import nn

from eurosat import data
from eurosat.flops import Count, counting
from eurosat.leaderboard import CURVE_KEYS, MAX_CURVE_POINTS, readable_bytes

TIME_LIMIT_SECONDS = 10 * 60
# Counting slows training down 2 to 6 times, so the counting run gets more time.
COUNT_LIMIT_SECONDS = 30 * 60

type TrainFunction = Callable[..., nn.Module]


class SubmissionError(Exception):
    """Something is wrong with a submission. The message says what, in plain words."""


@dataclass
class Result:
    name: str
    parameters: int
    train_seconds: float
    date: str
    code_url: str | None = None
    model_bytes: int | None = None  # memory the returned model holds on to
    curve: list[dict] = field(default_factory=list)  # what train() passed to log()
    train_flops: int | None = None  # None when compute was not counted
    predict_flops: int | None = None  # per test image; None when not counted
    uncounted_ops: list[str] | None = None  # heavy-looking ops the counter skipped
    predictions: str | None = None  # official runs: one digit per test image
    accuracy: float | None = None  # local runs only: accuracy on the held-back images


def load_train_function(folder: Path) -> TrainFunction:
    """Import model.py from a submission folder and return its train() function."""
    path = folder / "model.py"
    if not path.is_file():
        raise SubmissionError(f"{path} not found. Each submission needs a model.py file.")

    spec = importlib.util.spec_from_file_location(f"submission_{folder.name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train = getattr(module, "train", None)
    if not callable(train):
        raise SubmissionError(f"{path} must define: def train(train_set, log) -> nn.Module")
    return train


class Logger:
    """The `log` function train() receives. Each call adds one point to the curves."""

    def __init__(self) -> None:
        self.points: list[dict] = []

    def __call__(self, **metrics: float) -> None:
        unknown = set(metrics) - set(CURVE_KEYS)
        if unknown:
            raise SubmissionError(
                f"log() got {', '.join(sorted(unknown))}. It takes: {', '.join(CURVE_KEYS)}."
            )
        if len(self.points) >= MAX_CURVE_POINTS:
            return  # keep the first points; the curve is for looking, not for scoring
        point = {}
        for key, value in metrics.items():
            value = float(value)
            if math.isfinite(value):
                point[key] = round(value, 4)
        self.points.append(point)


@contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Stop the code inside if it runs longer than `seconds`.

    This uses an alarm signal, which exists on Linux and macOS. On Windows the
    code is not stopped early, but train_with_time_limit() still rejects it.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def stop(signum, frame):
        raise SubmissionError(f"Training took longer than the limit of {seconds:.0f} seconds.")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def call_train(train: TrainFunction, train_set: data.EuroSAT, log: Logger) -> nn.Module:
    """Call train(train_set, log), or train(train_set) if it takes no log."""
    try:
        takes_log = len(inspect.signature(train).parameters) >= 2
    except TypeError, ValueError:
        takes_log = True
    return train(train_set, log) if takes_log else train(train_set)


def train_with_time_limit(
    train: TrainFunction, train_set: data.EuroSAT, limit: float = TIME_LIMIT_SECONDS
) -> tuple[nn.Module, float, list[dict]]:
    """Run the student's train(). Return the model, its seconds and its logged curve."""
    torch.manual_seed(0)  # same starting point for everyone, so runs are repeatable
    log = Logger()
    start = time.perf_counter()
    with time_limit(limit):
        model = call_train(train, train_set, log)
    seconds = time.perf_counter() - start

    if seconds > limit:
        raise SubmissionError(f"Training took {seconds:.0f} seconds. The limit is {limit:.0f}.")
    if not isinstance(model, nn.Module):
        raise SubmissionError("train() must return a PyTorch model (an nn.Module).")
    return model, seconds, log.points


def model_bytes(model: nn.Module) -> int:
    """How much memory the model holds on to, in bytes.

    Parameters count, and so do buffers and any other tensor the model keeps: in a
    plain attribute, a list, a dict, or another object it points to. A model that
    stores the training images is as big as those images. Tensors that share memory
    are counted once.
    """
    seen_objects: set[int] = set()
    seen_memory: set[tuple[str, int]] = set()
    total = 0
    todo: list[object] = [model]
    while todo:
        obj = todo.pop()
        if id(obj) in seen_objects:
            continue
        seen_objects.add(id(obj))
        if isinstance(obj, torch.Tensor):
            try:
                storage = obj.untyped_storage()
                key, size = (str(storage.device), storage.data_ptr()), storage.nbytes()
            except RuntimeError, NotImplementedError:  # sparse tensors have no single storage
                key, size = (str(obj.device), id(obj)), obj.numel() * obj.element_size()
            if key not in seen_memory:
                seen_memory.add(key)
                total += size
        elif isinstance(obj, dict):
            todo.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            todo.extend(obj)
        elif hasattr(obj, "__dict__") and not isinstance(
            obj, (type, ModuleType, FunctionType, MethodType, BuiltinFunctionType)
        ):
            todo.extend(vars(obj).values())
    return total


def count_parameters(model: nn.Module) -> int:
    """How many numbers the model learns by gradient descent (its parameters).

    Buffers, such as stored averages, are not counted. A model can have 0.
    """
    return sum(p.numel() for p in model.parameters())


def count_flops(
    train: TrainFunction, train_set: data.EuroSAT, limit: float = COUNT_LIMIT_SECONDS
) -> Count | None:
    """Train again inside PyTorch's FLOP counter and return the training compute.

    The counter adds up the floating-point operations of every matrix multiply and
    convolution, forward and backward, including any validation inside train(). It
    skips cheap elementwise work such as ReLU or optimizer steps. The submission's
    code is not changed: the counter only watches the operations as they run.

    Counting slows training down, so this is a separate run from the timed one, with
    its own limit. If it fails, the score is kept and has no compute (None).
    """
    try:
        with counting() as count:
            train_with_time_limit(train, train_set, limit)
    except Exception as error:  # the timed run worked, so keep its score whatever happens
        print(f"Could not count the compute: {error}", file=sys.stderr)
        return None
    return count


def readable_flops(flops: int) -> str:
    """225000000000 -> "225 GFLOP", the same short form as the leaderboard page."""
    for suffix, size in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if flops >= size:
            value = flops / size
            return f"{value:.1f} {suffix}FLOP" if value < 10 else f"{value:.0f} {suffix}FLOP"
    return f"{flops} FLOP"


def predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """The model's answer (a class number) for each uint8 image."""
    model = model.to("cpu").eval()
    answers = []
    with torch.inference_mode():
        for batch in images.split(500):
            outputs = model(batch.float() / 255)
            expected = (len(batch), len(data.CLASSES))
            if not isinstance(outputs, torch.Tensor) or tuple(outputs.shape) != expected:
                shape = tuple(outputs.shape) if isinstance(outputs, torch.Tensor) else outputs
                raise SubmissionError(
                    f"For a batch of {len(batch)} images the model returned {shape}. "
                    f"It must return one row of {len(data.CLASSES)} class scores per image, "
                    f"shape {expected}."
                )
            answers.append(outputs.argmax(dim=1))
    return torch.cat(answers)


def code_url(name: str) -> str | None:
    """A link to the submission's code on GitHub, when running inside GitHub Actions."""
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    if not (server and repo and sha):
        return None
    return f"{server}/{repo}/blob/{sha}/submissions/{name}/model.py"


@dataclass
class Scored:
    result: Result
    model: nn.Module
    check_set: data.EuroSAT | None  # local runs: the held-back images and labels
    predicted: torch.Tensor  # answers for check_set (local) or the test images (official)


def score(
    folder: Path,
    official: bool = False,
    flops: bool = False,
    quick: bool = False,
    limit: float = TIME_LIMIT_SECONDS,
) -> Scored:
    """Train one submission and measure it. With flops=True, also count its compute."""
    train = load_train_function(folder)
    # Load the data before training: downloading the first time is not training time,
    # and the test images are in memory before student code runs.
    full = data.train_set()
    if official:
        train_set, check_set = full, None
        images = data.test_set().images
    else:
        train_set, check_set = data.split_for_check(full)
        if quick:
            train_set = train_set.subset(torch.arange(len(train_set) // 4))
        images = check_set.images

    model, seconds, curve = train_with_time_limit(train, train_set, limit)
    if flops:
        # Labelling images is counted too: a model can be cheap to train and costly to
        # use, like one that compares each image with every training image.
        with counting() as counted:
            predicted = predict(model, images)
    else:
        predicted = predict(model, images)

    result = Result(
        name=folder.name,
        parameters=count_parameters(model),
        train_seconds=round(seconds, 1),
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        code_url=code_url(folder.name),
        model_bytes=model_bytes(model),
        curve=curve,
    )
    if official:
        result.predictions = "".join(str(int(p)) for p in predicted)
    else:
        assert check_set is not None and check_set.labels is not None
        result.accuracy = round((predicted == check_set.labels).float().mean().item(), 4)
    if flops:
        result.predict_flops = round(counted.flops / len(images))
        trained = count_flops(train, train_set)
        uncounted = counted.uncounted
        if trained is not None:
            result.train_flops = trained.flops
            uncounted = trained.uncounted + uncounted
        result.uncounted_ops = sorted(uncounted)
    return Scored(result, model, check_set, predicted)


def class_report(labels: torch.Tensor, predicted: torch.Tensor) -> str:
    """Accuracy of each class, and the mix-ups the model makes most often."""
    lines = ["Accuracy by class:"]
    for k, name in enumerate(data.CLASSES):
        mask = labels == k
        if not mask.any():
            continue
        share = (predicted[mask] == k).float().mean().item()
        bar = "#" * round(share * 20)
        lines.append(f"  {name:22} {share:6.1%}  {bar}")
    confusion = torch.zeros(len(data.CLASSES), len(data.CLASSES), dtype=torch.long)
    confusion.index_put_((labels, predicted), torch.ones_like(labels), accumulate=True)
    confusion.fill_diagonal_(0)
    top = confusion.flatten().topk(3)
    lines.append("Most common mix-ups (true -> answered):")
    for count, index in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        if count:
            true, answer = divmod(index, len(data.CLASSES))
            lines.append(f"  {data.CLASSES[true]} -> {data.CLASSES[answer]}: {count} images")
    return "\n".join(lines)


def save_report(scored: Scored, out: Path) -> list[Path]:
    """Save pictures that help you see what the model learned and where it fails."""
    from torchvision.utils import save_image  # only needed here

    out.mkdir(parents=True, exist_ok=True)
    saved = []
    conv = next(
        (m for m in scored.model.modules() if isinstance(m, nn.Conv2d) and m.in_channels == 3),
        None,
    )
    if conv is not None:
        # Each filter of the first layer as a tiny color picture: what it responds to.
        weights = conv.weight.detach().cpu().float()
        low = weights.amin(dim=(1, 2, 3), keepdim=True)
        high = weights.amax(dim=(1, 2, 3), keepdim=True)
        weights = (weights - low) / (high - low + 1e-8)
        size = max(32, 8 * weights.shape[-1])
        weights = nn.functional.interpolate(weights, size=size, mode="nearest")
        path = out / "filters.png"
        save_image(weights, path, nrow=8, padding=2, pad_value=1)
        saved.append(path)

    check = scored.check_set
    if check is not None and check.labels is not None:
        wrong = (scored.predicted != check.labels).nonzero().flatten()[:64]
        if len(wrong):
            path = out / "mistakes.png"
            save_image(check.images[wrong].float() / 255, path, nrow=8, padding=2, pad_value=1)
            text = out / "mistakes.txt"
            text.write_text(
                "Picture by picture, left to right and top to bottom (true -> answered):\n"
                + "".join(
                    f"{i + 1:3}. {data.CLASSES[check.labels[j]]} -> "
                    f"{data.CLASSES[scored.predicted[j]]}\n"
                    for i, j in enumerate(wrong.tolist())
                )
            )
            saved += [path, text]
    return saved


def save(result: Result, results_dir: Path) -> Path:
    """Write the result as a small JSON file, one file per scored submission."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{result.name}.json"
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return path


def print_result(scored: Scored) -> None:
    result = scored.result
    print()
    print(f"Submission:     {result.name}")
    if result.accuracy is not None:
        print(f"Check accuracy: {result.accuracy:.2%} (on held-back training images)")
    print(f"Parameters:     {result.parameters:,}")
    print(f"Model size:     {readable_bytes(result.model_bytes or 0)}")
    print(f"Training time:  {result.train_seconds:.0f} seconds")
    if result.train_flops is not None:
        print(f"Compute:        {readable_flops(result.train_flops)} to train")
    if result.predict_flops is not None:
        print(f"                {readable_flops(result.predict_flops)} per image")
    if result.uncounted_ops:
        print(
            f"Not counted:    {', '.join(result.uncounted_ops)}\n"
            "                These ran but have no FLOP formula. If they do heavy math,\n"
            "                the compute above is too low."
        )
    if scored.check_set is not None and scored.check_set.labels is not None:
        print(class_report(scored.check_set.labels, scored.predicted))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train submissions and print their scores.")
    parser.add_argument("folders", type=Path, nargs="+", help="for example: submissions/baseline")
    parser.add_argument("--quick", action="store_true", help="train on a quarter of the images")
    parser.add_argument(
        "--flops",
        action="store_true",
        help="also count the compute (FLOPs) of training and of labelling each image; "
        "trains each submission a second time",
    )
    parser.add_argument("--report", type=Path, metavar="DIR", help="save pictures here")
    parser.add_argument(
        "--official",
        action="store_true",
        help="what GitHub runs: train on all images and answer the secret test images",
    )
    parser.add_argument("--save", type=Path, metavar="DIR", help="also save the results here")
    args = parser.parse_args()
    if args.official and args.quick:
        parser.error("--quick and --official do not go together")

    failed = []
    for folder in args.folders:
        try:
            scored = score(folder, official=args.official, flops=args.flops, quick=args.quick)
        except SubmissionError as error:
            print(f"Error in {folder}: {error}", file=sys.stderr)
            failed.append(folder.name)
            continue
        except Exception:  # a crash in one submission must not stop the others
            print(f"Error in {folder}: its code crashed.", file=sys.stderr)
            traceback.print_exc()
            failed.append(folder.name)
            continue

        print_result(scored)
        if args.report:
            for path in save_report(scored, args.report / folder.name):
                print(f"Saved:          {path}")
        if args.save:
            print(f"Saved to:       {save(scored.result, args.save)}")

    if failed:
        sys.exit(f"Could not score: {', '.join(failed)}")
