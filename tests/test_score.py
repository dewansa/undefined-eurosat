"""Tests for the scoring script. They use tiny fake images, so they run in seconds."""

import json
import time
from pathlib import Path

import pytest
import torch
from torch import nn

from eurosat.data import EuroSAT, split_for_check
from eurosat.leaderboard import check_upload
from eurosat.score import (
    Logger,
    Result,
    Scored,
    SubmissionError,
    class_report,
    count_flops,
    count_parameters,
    counting,
    load_train_function,
    model_bytes,
    predict,
    readable_flops,
    save,
    save_report,
    train_with_time_limit,
)

SUBMISSIONS = sorted(p for p in Path("submissions").iterdir() if p.is_dir())
SIDE = 8  # fake images are 3x8x8, so the tests are fast


def fake_images(count: int = 20) -> EuroSAT:
    images = torch.randint(0, 256, (count, 3, SIDE, SIDE), dtype=torch.uint8)
    return EuroSAT(images, torch.arange(count) % 10)


def linear() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * SIDE * SIDE, 10))


def write_submission(folder: Path, code: str) -> Path:
    folder.mkdir()
    (folder / "model.py").write_text(code)
    return folder


@pytest.mark.parametrize("folder", SUBMISSIONS, ids=lambda p: p.name)
def test_every_submission_has_a_train_function(folder: Path) -> None:
    assert callable(load_train_function(folder))


def test_missing_model_file(tmp_path: Path) -> None:
    with pytest.raises(SubmissionError, match="model.py"):
        load_train_function(tmp_path)


def test_missing_train_function(tmp_path: Path) -> None:
    folder = write_submission(tmp_path / "alice", "x = 1\n")
    with pytest.raises(SubmissionError, match="def train"):
        load_train_function(folder)


def test_train_must_return_a_model() -> None:
    with pytest.raises(SubmissionError, match="nn.Module"):
        train_with_time_limit(lambda train_set, log: "not a model", fake_images())


def test_train_without_log_still_works() -> None:
    model, _, curve = train_with_time_limit(lambda train_set: linear(), fake_images())
    assert isinstance(model, nn.Module)
    assert curve == []


def test_logged_points_become_the_curve() -> None:
    def train(train_set, log):
        for epoch in range(3):
            log(train_loss=1 / (epoch + 1), valid_accuracy=0.5 + epoch / 10)
        return linear()

    _, _, curve = train_with_time_limit(train, fake_images())
    assert curve == [
        {"train_loss": 1.0, "valid_accuracy": 0.5},
        {"train_loss": 0.5, "valid_accuracy": 0.6},
        {"train_loss": 0.3333, "valid_accuracy": 0.7},
    ]


def test_log_rejects_unknown_names_and_keeps_tensors_as_numbers() -> None:
    log = Logger()
    log(train_loss=torch.tensor(0.25))
    assert log.points == [{"train_loss": 0.25}]
    with pytest.raises(SubmissionError, match="validation_acc"):
        log(validation_acc=0.9)


def test_training_that_runs_too_long_is_stopped() -> None:
    def slow_train(train_set, log):
        time.sleep(5)
        return linear()

    start = time.perf_counter()
    with pytest.raises(SubmissionError, match="longer than"):
        train_with_time_limit(slow_train, fake_images(), limit=0.5)
    assert time.perf_counter() - start < 2


def test_predict_gives_one_class_per_image() -> None:
    model = linear()
    assert count_parameters(model) == 3 * SIDE * SIDE * 10 + 10
    answers = predict(model, fake_images().images)
    assert answers.shape == (20,)
    assert answers.min() >= 0 and answers.max() < 10


@pytest.mark.parametrize("shape", [(4, 10, 1), (4, 1), (4, 9), (3, 10)])
def test_wrong_output_shape_is_an_error_not_a_score(shape: tuple[int, ...]) -> None:
    # (4, 1) used to score: every answer became class 0.
    class Wrong(nn.Module):
        def forward(self, images):
            return torch.zeros(shape)

    with pytest.raises(SubmissionError, match=r"\(4, 10\)"):
        predict(Wrong(), fake_images(4).images)


def test_output_must_be_a_tensor() -> None:
    class Words(nn.Module):
        def forward(self, images):
            return ["forest"] * len(images)

    with pytest.raises(SubmissionError, match="class scores"):
        predict(Words(), fake_images(4).images)


def test_the_check_split_is_always_the_same_and_does_not_overlap() -> None:
    full = EuroSAT(torch.arange(100, dtype=torch.uint8).view(100, 1, 1, 1), torch.arange(100))
    train_a, check_a = split_for_check(full)
    train_b, check_b = split_for_check(full)
    assert torch.equal(check_a.labels, check_b.labels)
    assert len(check_a) == 20 and len(train_a) == 80
    assert not set(check_a.labels.tolist()) & set(train_a.labels.tolist())


def one_training_step(train_set, log):
    images, labels = train_set.images.float() / 255, train_set.labels
    model = linear()
    nn.functional.cross_entropy(model(images), labels).backward()
    return model


def test_flops_of_one_training_step() -> None:
    # Forward: 20 images times a 192x10 matrix, 2 FLOPs (multiply, add) per weight.
    # Backward: the same again for the weight gradient. The images need no gradient.
    forward = 2 * 20 * 3 * SIDE * SIDE * 10
    count = count_flops(one_training_step, fake_images(20))
    assert count is not None
    assert count.flops == 2 * forward
    assert not count.uncounted  # a plain training step has nothing heavy left uncounted


def test_prediction_compute_is_counted() -> None:
    with counting() as count:
        predict(linear(), fake_images(20).images)
    assert count.flops == 2 * 20 * 3 * SIDE * SIDE * 10


def test_ops_without_a_formula_are_reported() -> None:
    with counting() as count:
        torch.linalg.matrix_exp(torch.rand(10, 10))
        torch.rand(4, 10).relu().sum()
    assert set(count.uncounted) == {"aten.linalg_matrix_exp"}


def test_model_size_counts_every_tensor_the_model_keeps() -> None:
    class Memorizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(10, 2)  # 22 numbers
            self.register_buffer("kept", torch.zeros(100))  # 100 numbers
            self.images = torch.zeros(1000)  # plain attribute: 1000 numbers
            self.extra = {"rows": [torch.zeros(50, dtype=torch.uint8)]}  # 50 bytes
            self.same = self.images[:10]  # shares memory with images: free

    assert model_bytes(Memorizer()) == 4 * (22 + 100 + 1000) + 50


def test_counting_does_not_change_the_model() -> None:
    plain, _, _ = train_with_time_limit(one_training_step, fake_images())
    count_flops(one_training_step, fake_images())
    counted, _, _ = train_with_time_limit(one_training_step, fake_images())
    assert torch.equal(plain[1].weight, counted[1].weight)


def test_flops_are_none_when_counting_fails() -> None:
    def slow_train(train_set, log):
        time.sleep(5)
        return linear()

    calls = []

    def crashes_the_second_time(train_set, log):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("out of memory")
        return linear()

    assert count_flops(slow_train, fake_images(), limit=0.5) is None
    train_with_time_limit(crashes_the_second_time, fake_images())
    assert count_flops(crashes_the_second_time, fake_images()) is None


def test_readable_flops() -> None:
    assert readable_flops(0) == "0 FLOP"
    assert readable_flops(224_900_000_000) == "225 GFLOP"
    assert readable_flops(3_272_000_000_000) == "3.3 TFLOP"


def test_class_report_names_the_mix_ups() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    report = class_report(labels, torch.tensor([0, 1, 1, 1]))
    assert "Annual crop" in report and "50.0%" in report
    assert "Annual crop -> Forest: 1 images" in report


def test_report_saves_filters_and_mistakes(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Flatten(), nn.LazyLinear(10))
    check = fake_images(8)
    model(check.images.float())
    scored = Scored(Result("alice", 0, 1.0, "now"), model, check, (check.labels + 1) % 10)
    saved = save_report(scored, tmp_path)
    assert {p.name for p in saved} == {"filters.png", "mistakes.png", "mistakes.txt"}
    assert "Annual crop -> Forest" in (tmp_path / "mistakes.txt").read_text()


def test_official_result_file_is_accepted_by_the_grader(tmp_path: Path) -> None:
    # What the pull request check saves must pass the grader's checks.
    result = Result(
        "alice",
        1000,
        12.3,
        "2026-01-02T03:04:05+00:00",
        model_bytes=4000,
        curve=[{"train_loss": 0.5}],
        predictions="0123",
    )
    path = save(result, tmp_path)
    assert path.name == "alice.json"
    trusted = check_upload(json.loads(path.read_text()), "alice", 4, reviewed=False)
    assert trusted["model_bytes"] == 4000
    assert trusted["curve"] == [{"train_loss": 0.5}]
