"""Tests for grading and the leaderboard rows, above all the checks on uploaded answers."""

import json
from pathlib import Path

import pytest

from eurosat import data
from eurosat.leaderboard import (
    CLASSES,
    InvalidResult,
    add_pr_rows,
    add_reviewed_rows,
    all_rows,
    auto_merge_reason,
    check_upload,
    comment,
    grade,
    grade_uploads,
    margin,
    readable_bytes,
    remove_closed_pr_rows,
    remove_pr_rows,
)

LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
GOOD = {
    "name": "alice",
    "parameters": 1000,
    "train_seconds": 42.0,
    "date": "2026-01-02T03:04:05+00:00",
    "code_url": None,
    "model_bytes": 4000,
    "curve": [{"train_loss": 1.5, "valid_accuracy": 0.5}],
    "predictions": "0123456789",
    "train_flops": None,
    "predict_flops": None,
    "uncounted_ops": None,
}


def test_classes_match_the_data_loader() -> None:
    # leaderboard.py runs without PyTorch, so it keeps its own copy.
    assert CLASSES == data.CLASSES


def test_good_upload_is_kept() -> None:
    trusted = check_upload(GOOD, "alice", 10, reviewed=False)
    assert trusted == {
        "name": "alice",
        "predictions": "0123456789",
        "parameters": 1000,
        "train_seconds": 42.0,
        "model_bytes": 4000,
        "curve": [{"train_loss": 1.5, "valid_accuracy": 0.5}],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"name": "bob"},
        {"predictions": "012345678"},  # one answer short
        {"predictions": "012345678x"},
        {"predictions": 123},
        {"parameters": -1},
        {"parameters": 1.5},
        {"train_seconds": 601},
        {"train_seconds": float("nan")},
        {"model_bytes": -5},
        {"train_flops": 10},  # compute is only counted after the merge
        {"accuracy_override": 1.0},  # unknown field
        {"curve": [{"test_accuracy": 1.0}]},
        {"curve": [{"train_loss": "low"}]},
        {"curve": [{"train_loss": 1.0}] * 501},
    ],
)
def test_bad_upload_is_rejected(change: dict) -> None:
    with pytest.raises(InvalidResult):
        check_upload({**GOOD, **change}, "alice", 10, reviewed=False)


def test_reviewed_upload_keeps_compute_and_checks_the_code_link() -> None:
    url = "https://github.com/me/repo/blob/" + "a" * 40 + "/submissions/alice/model.py"
    raw = {**GOOD, "code_url": url, "train_flops": 10, "predict_flops": 2, "uncounted_ops": []}
    trusted = check_upload(raw, "alice", 10, reviewed=True)
    assert trusted["train_flops"] == 10
    assert trusted["code_url"] == url
    with pytest.raises(InvalidResult, match="code link"):
        check_upload({**raw, "code_url": "javascript:alert(1)"}, "alice", 10, reviewed=True)
    with pytest.raises(InvalidResult, match="operation names"):
        check_upload({**raw, "uncounted_ops": ["<b>"]}, "alice", 10, reviewed=True)


def test_grade_counts_accuracy_by_class_and_mix_ups() -> None:
    graded = grade("0123456788", LABELS)
    assert graded["accuracy"] == 0.9
    assert graded["class_accuracy"][9] == 0
    assert graded["class_accuracy"][0] == 1
    assert graded["mixups"] == [[9, 8, 1]]
    assert graded["test_images"] == 10


def test_grade_uploads_reports_missing_and_bad_files(tmp_path: Path) -> None:
    (tmp_path / "alice.json").write_text(json.dumps(GOOD))
    (tmp_path / "bob.json").write_text("{not json")
    rows, failed = grade_uploads(tmp_path, ["alice", "bob", "carol"], LABELS, reviewed=False)
    assert [r["name"] for r in rows] == ["alice"]
    assert rows[0]["accuracy"] == 1.0
    assert "predictions" not in rows[0]  # answers are never published
    assert set(failed) == {"bob", "carol"}
    assert "could not be trained" in failed["carol"]


def test_pr_rows_replace_earlier_ones_and_go_away(tmp_path: Path) -> None:
    row = {"name": "alice", "accuracy": 0.5}
    add_pr_rows(tmp_path, 7, "https://github.com/x/y/pull/7", [row])
    add_pr_rows(tmp_path, 7, "https://github.com/x/y/pull/7", [{**row, "accuracy": 0.6}])
    rows = all_rows(tmp_path)
    assert [(r["accuracy"], r["reviewed"], r["pr"]) for r in rows] == [(0.6, False, 7)]

    remove_closed_pr_rows(tmp_path, open_prs={7})
    assert len(all_rows(tmp_path)) == 1
    remove_pr_rows(tmp_path, 7)
    assert all_rows(tmp_path) == []


def test_reviewed_rows_never_overwrite_each_other(tmp_path: Path) -> None:
    add_reviewed_rows(tmp_path, [{"name": "alice", "accuracy": 0.5}])
    (tmp_path / "2000-01-01T00-00-00_alice.json").write_text('{"name": "alice", "accuracy": 0.4}')
    add_reviewed_rows(tmp_path, [{"name": "alice", "accuracy": 0.6}])
    assert len(list(tmp_path.glob("*.json"))) >= 2
    assert all(r["reviewed"] for r in all_rows(tmp_path))


def test_comment_compares_with_the_baseline_and_your_best() -> None:
    graded = {**check_upload(GOOD, "alice", 10, reviewed=False), **grade("0123456788", LABELS)}
    graded.pop("predictions")
    earlier = [
        {"name": "baseline", "accuracy": 0.8, "class_accuracy": [0.5] * 10},
        {"name": "alice", "accuracy": 0.95},
    ]
    text = comment([graded], {"bob": "it could not be trained and scored"}, earlier, "URL")
    assert "90.00% test accuracy" in text
    assert "+10.0 points vs the baseline" in text
    assert "-5.0 vs your best" in text
    assert "| Sea or lake | 0% | 50% |" in text
    assert "Sea or lake → Permanent crop" not in text and "Sea or lake → River (1)" in text
    assert "**bob**" in text


def test_margin_shrinks_with_more_test_images() -> None:
    assert round(margin(0.9, 5400) * 100, 1) == 0.8
    assert margin(0.9, 100) > margin(0.9, 5400)
    assert margin(1.0, 5400) > 0  # never claims to be exact


@pytest.mark.parametrize(
    ("files", "allowed"),
    [
        (["submissions/alice/model.py"], True),
        (["submissions/Alice-v2/model.py", "submissions/alice/notes.md"], True),
        (["submissions/alice/model.py", "README.md"], False),
        (["submissions/bob/model.py"], False),
        (["submissions/alicex/model.py"], False),
        ([], False),
    ],
)
def test_auto_merge_only_for_the_authors_own_folders(files: list[str], allowed: bool) -> None:
    assert (auto_merge_reason("alice", files) is None) is allowed


def test_readable_bytes() -> None:
    assert readable_bytes(203560) == "204 kB"
    assert readable_bytes(5) == "5 B"
