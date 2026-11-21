"""Grade uploaded answers and keep the leaderboard's rows. Uses only the standard library.

The job that trains a submission runs student code, so it never sees the test labels.
It uploads the model's answers for the test images. This file, run by a job that never
runs student code, checks each upload strictly, compares the answers with the secret
labels (the TEST_LABELS repo secret) and writes the rows.

There are two kinds of rows:

- Reviewed: results/<date>_<name>.json. The submission was merged into main and
  scored there, with its compute counted. Every merged submission keeps its own row.
- Not reviewed: results/pending/pr-<number>_<name>.json. Scored on an open pull
  request. Each new push replaces the pull request's rows, and they go away when the
  pull request is merged or closed.

GitHub Actions runs this file directly (python src/eurosat/leaderboard.py ...),
without installing PyTorch.
"""

import argparse
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,38}")
CLASSES = (
    "Annual crop",
    "Forest",
    "Herbaceous vegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "Permanent crop",
    "Residential",
    "River",
    "Sea or lake",
)
CURVE_KEYS = ("train_loss", "valid_loss", "train_accuracy", "valid_accuracy")
MAX_CURVE_POINTS = 500
MAX_FILE_BYTES = 200_000
MAX_TRAIN_SECONDS = 10 * 60
MAX_PARAMETERS = 1_000_000_000
MAX_MODEL_BYTES = 100_000_000_000
MAX_FLOPS = 10**18
REQUIRED_KEYS = {"name", "parameters", "train_seconds", "predictions"}
OPTIONAL_KEYS = {"date", "code_url", "model_bytes", "curve", "accuracy"}
COMPUTE_KEYS = {"train_flops", "predict_flops", "uncounted_ops"}
CODE_URL = re.compile(
    r"https://github\.com/[\w.-]+/[\w.-]+/blob/[0-9a-f]{40}/submissions/[\w.-]+/model\.py"
)
OP_NAME = re.compile(r"[A-Za-z0-9_.]{1,80}")


class InvalidResult(Exception):
    """An uploaded result that does not look right. It is not put on the leaderboard."""


def is_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def check_upload(raw: object, name: str, n_test: int, reviewed: bool) -> dict:
    """Check one uploaded result and return only the trusted fields.

    Compute (FLOPs) is counted only after a merge, so a pull request upload that
    reports compute has been tampered with.
    """
    allowed = REQUIRED_KEYS | OPTIONAL_KEYS | COMPUTE_KEYS
    if not isinstance(raw, dict) or not REQUIRED_KEYS <= set(raw) <= allowed:
        raise InvalidResult("the result does not have the expected fields")
    if raw["name"] != name:
        raise InvalidResult(f"the result is for {raw['name']!r}, not {name!r}")
    if not reviewed and any(raw.get(key) is not None for key in COMPUTE_KEYS):
        raise InvalidResult("compute is counted after the merge, not on pull requests")

    predictions = raw["predictions"]
    if not isinstance(predictions, str) or not re.fullmatch(r"[0-9]*", predictions):
        raise InvalidResult("the answers must be one digit per test image")
    if len(predictions) != n_test:
        raise InvalidResult(f"there are {len(predictions)} answers for {n_test} test images")

    parameters = raw["parameters"]
    if type(parameters) is not int or not 0 <= parameters <= MAX_PARAMETERS:
        raise InvalidResult("parameters must be a whole number, 0 or more")
    seconds = raw["train_seconds"]
    if not is_number(seconds) or not 0 <= seconds <= MAX_TRAIN_SECONDS:
        raise InvalidResult(f"training time must be from 0 to {MAX_TRAIN_SECONDS} seconds")

    trusted = {
        "name": name,
        "predictions": predictions,
        "parameters": parameters,
        "train_seconds": round(float(seconds), 1),
        "curve": check_curve(raw.get("curve")),
    }
    size = raw.get("model_bytes")
    if size is not None:
        if type(size) is not int or not 0 <= size <= MAX_MODEL_BYTES:
            raise InvalidResult("model size must be a whole number of bytes, 0 or more")
        trusted["model_bytes"] = size
    if reviewed:
        url = raw.get("code_url")
        if url is not None and not (isinstance(url, str) and CODE_URL.fullmatch(url)):
            raise InvalidResult("the code link does not look right")
        trusted["code_url"] = url
        for key in ("train_flops", "predict_flops"):
            value = raw.get(key)
            if value is not None and (type(value) is not int or not 0 <= value <= MAX_FLOPS):
                raise InvalidResult(f"{key} must be a whole number, 0 or more")
            trusted[key] = value
        ops = raw.get("uncounted_ops") or []
        if not isinstance(ops, list) or len(ops) > 50:
            raise InvalidResult("uncounted_ops must be a short list")
        if not all(isinstance(op, str) and OP_NAME.fullmatch(op) for op in ops):
            raise InvalidResult("uncounted_ops must be operation names")
        trusted["uncounted_ops"] = ops
    return trusted


def check_curve(curve: object) -> list[dict]:
    """The points train() logged: up to MAX_CURVE_POINTS, each with known keys only."""
    if curve is None:
        return []
    if not isinstance(curve, list) or len(curve) > MAX_CURVE_POINTS:
        raise InvalidResult("the training curve must be a list of points")
    points = []
    for point in curve:
        if not isinstance(point, dict) or not set(point) <= set(CURVE_KEYS):
            raise InvalidResult(f"curve points may only have: {', '.join(CURVE_KEYS)}")
        if not all(is_number(v) and abs(v) < 1e6 for v in point.values()):
            raise InvalidResult("curve values must be numbers")
        points.append({k: round(float(v), 4) for k, v in point.items()})
    return points


def read_labels(text: str) -> list[int]:
    labels = text.strip()
    if not labels or not re.fullmatch(r"[0-9]+", labels):
        raise SystemExit("The test labels are missing. Set the TEST_LABELS repo secret.")
    return [int(c) for c in labels]


def grade(predictions: str, labels: list[int]) -> dict:
    """Accuracy overall and by class, and the most common mix-ups."""
    n = len(CLASSES)
    confusion = [[0] * n for _ in range(n)]
    for answer, label in zip(predictions, labels, strict=True):
        confusion[label][int(answer)] += 1
    correct = sum(confusion[k][k] for k in range(n))
    by_class = [round(confusion[k][k] / max(1, sum(confusion[k])), 4) for k in range(n)]
    mixups = sorted(
        ([t, a, confusion[t][a]] for t in range(n) for a in range(n) if t != a and confusion[t][a]),
        key=lambda m: -m[2],
    )[:3]
    return {
        "accuracy": round(correct / len(labels), 4),
        "class_accuracy": by_class,
        "mixups": mixups,
        "test_images": len(labels),
    }


def grade_uploads(
    folder: Path, names: list[str], labels: list[int], reviewed: bool
) -> tuple[list[dict], dict[str, str]]:
    """Grade each expected submission's upload: (graded rows, {name: why it failed})."""
    rows, failed = [], {}
    for name in names:
        path = folder / f"{name}.json"
        try:
            if not NAME_PATTERN.fullmatch(name):
                raise InvalidResult("the folder name is not valid")
            if not path.is_file():
                raise InvalidResult("it could not be trained and scored")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise InvalidResult("the uploaded file is too big")
            trusted = check_upload(json.loads(path.read_text()), name, len(labels), reviewed)
        except (InvalidResult, ValueError) as error:
            failed[name] = str(error)
            continue
        predictions = trusted.pop("predictions")
        rows.append({**trusted, **grade(predictions, labels)})
    return rows, failed


def readable_bytes(size: int) -> str:
    """188160000 -> "188 MB", the same short form as the leaderboard page."""
    for suffix, unit in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if size >= unit:
            value = size / unit
            return f"{value:.1f} {suffix}" if value < 10 else f"{value:.0f} {suffix}"
    return f"{size} B"


def margin(accuracy: float, n: int) -> float:
    """About 95% of the time, luck moves a score by less than this (normal approximation)."""
    return 1.96 * math.sqrt(max(accuracy * (1 - accuracy), 1e-4) / n)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def add_pr_rows(results_dir: Path, pr: int, pr_url: str, rows: list[dict]) -> None:
    """Replace a pull request's not reviewed rows with new ones."""
    pending = results_dir / "pending"
    remove_pr_rows(results_dir, pr)
    pending.mkdir(parents=True, exist_ok=True)
    date = now()
    for row in rows:
        row = {**row, "date": date, "reviewed": False, "pr": pr, "code_url": pr_url}
        (pending / f"pr-{pr}_{row['name']}.json").write_text(json.dumps(row, indent=2) + "\n")


def add_reviewed_rows(results_dir: Path, rows: list[dict]) -> None:
    """Save merged submissions' rows. Each gets a fresh file, so none is overwritten."""
    results_dir.mkdir(parents=True, exist_ok=True)
    date = now()
    stamp = date.replace(":", "-").removesuffix("+00-00")
    for row in rows:
        path = results_dir / f"{stamp}_{row['name']}.json"
        path.write_text(json.dumps({**row, "date": date}, indent=2) + "\n")


def remove_pr_rows(results_dir: Path, pr: int) -> None:
    for path in (results_dir / "pending").glob(f"pr-{pr}_*.json"):
        path.unlink()


def remove_closed_pr_rows(results_dir: Path, open_prs: set[int]) -> None:
    """Remove not reviewed rows whose pull request was merged or closed."""
    for path in (results_dir / "pending").glob("pr-*_*.json"):
        number = int(path.name.removeprefix("pr-").split("_")[0])
        if number not in open_prs:
            path.unlink()


def reviewed_rows(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def all_rows(results_dir: Path) -> list[dict]:
    """Every row on the leaderboard, reviewed and not reviewed."""
    rows = [{**row, "reviewed": True} for row in reviewed_rows(results_dir)]
    for path in sorted((results_dir / "pending").glob("*.json")):
        rows.append({**json.loads(path.read_text()), "reviewed": False})
    return rows


def points(change: float) -> str:
    return f"{change * 100:+.1f}"


def comment(
    rows: list[dict], failed: dict[str, str], earlier: list[dict], leaderboard_url: str
) -> str:
    """The text of the bot's comment on the pull request."""
    lines = []
    if failed:
        lines.append("I could not score:")
        lines += [
            f"- **{name}**: {why}. Open the Check run's log to see why."
            for name, why in failed.items()
        ]
        lines.append("")
    if not rows:
        return "\n".join(lines or ["I could not score this pull request."])

    baseline = next((r for r in reversed(earlier) if r["name"] == "baseline"), None)
    for row in rows:
        best = max(
            (r for r in earlier if r["name"] == row["name"]),
            key=lambda r: r["accuracy"],
            default=None,
        )
        size = f", {readable_bytes(row['model_bytes'])}" if "model_bytes" in row else ""
        lines.append(
            f"### {row['name']}: {row['accuracy']:.2%} test accuracy "
            f"(± {margin(row['accuracy'], row['test_images']) * 100:.1f})"
        )
        compare = []
        if baseline and row["name"] != "baseline":
            compare.append(
                f"{points(row['accuracy'] - baseline['accuracy'])} points vs the baseline"
            )
        if best:
            compare.append(
                f"{points(row['accuracy'] - best['accuracy'])} vs your best merged score"
            )
        lines.append(
            f"{row['parameters']:,} parameters{size}, "
            f"{row['train_seconds']:.0f} seconds of training"
            + (f". {'; '.join(compare)}." if compare else ".")
        )
        lines += [
            "",
            "| Class | Yours |" + (" Baseline |" if baseline else ""),
            "|---|---:|" + ("---:|" if baseline else ""),
        ]
        for k, name in enumerate(CLASSES):
            cell = f"| {name} | {row['class_accuracy'][k]:.0%} |"
            if baseline and "class_accuracy" in baseline:
                cell += f" {baseline['class_accuracy'][k]:.0%} |"
            lines.append(cell)
        if row["mixups"]:
            mix = "; ".join(f"{CLASSES[t]} → {CLASSES[a]} ({c})" for t, a, c in row["mixups"])
            lines += ["", f"Most common mix-ups (true → answered): {mix}."]
        if not row["curve"]:
            lines += [
                "",
                "Tip: call `log(train_loss=..., valid_accuracy=...)` once per epoch "
                "to see your training curves on the leaderboard.",
            ]
        lines.append("")
    lines.append(
        f"It is on the [leaderboard]({leaderboard_url}) as *not reviewed*. After the merge it "
        "is scored again on main, its compute (FLOPs) is counted, and it becomes reviewed. "
        "The ± is how far luck alone can move a score on 5,400 test images; smaller "
        "differences may not be real."
    )
    return "\n".join(lines)


def auto_merge_reason(author: str, files: list[str]) -> str | None:
    """Why a pull request may NOT be merged automatically, or None when it may.

    It may when every changed file is inside one or more submissions/<folder>/ named
    after the author: <github-name> or <github-name>-<anything>.
    """
    if not files:
        return "it changes no files"
    me = author.lower()
    for path in files:
        parts = path.split("/")
        if len(parts) < 3 or parts[0] != "submissions":
            return f"{path} is outside submissions/"
        folder = parts[1].lower()
        if folder != me and not folder.startswith(me + "-"):
            return f"{path} is not in a folder named after {author}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade uploads and update the leaderboard.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add-pr", help="grade a pull request's uploads as not reviewed")
    add.add_argument("--pr", type=int, required=True)
    add.add_argument("--pr-url", required=True)
    add.add_argument("--uploaded", type=Path, required=True)
    add.add_argument("--names", required=True, help="submission folders the PR changes")
    add.add_argument("--leaderboard-url", required=True)
    add.add_argument("--comment-file", type=Path, required=True)
    add.add_argument("--ok-file", type=Path, help="written with 'true' when every name scored")

    merged = commands.add_parser("add-reviewed", help="grade merged submissions' uploads")
    merged.add_argument("--uploaded", type=Path, required=True)
    merged.add_argument("--names", required=True)

    clean = commands.add_parser("clean", help="remove rows of merged or closed pull requests")
    clean.add_argument("--open", default="", help="numbers of the open pull requests")
    clean.add_argument("--pr", type=int, help="also remove this open pull request's rows")

    check = commands.add_parser("auto-merge", help="say whether a pull request may auto-merge")
    check.add_argument("--author", required=True)
    check.add_argument("--files", type=Path, required=True, help="changed paths, one per line")

    build = commands.add_parser("build", help="write every row into one file for the website")
    build.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    match args.command:
        case "add-pr":
            labels = read_labels(os.environ.get("TEST_LABELS", ""))
            names = args.names.split()
            rows, failed = grade_uploads(args.uploaded, names, labels, reviewed=False)
            add_pr_rows(args.results, args.pr, args.pr_url, rows)
            earlier = reviewed_rows(args.results)
            args.comment_file.write_text(
                comment(rows, failed, earlier, args.leaderboard_url) + "\n"
            )
            if args.ok_file:
                args.ok_file.write_text("true" if rows and not failed else "false")
        case "add-reviewed":
            labels = read_labels(os.environ.get("TEST_LABELS", ""))
            rows, failed = grade_uploads(args.uploaded, args.names.split(), labels, reviewed=True)
            for name, why in failed.items():
                print(f"::warning::Could not score {name}: {why}")
            add_reviewed_rows(args.results, rows)
        case "clean":
            remove_closed_pr_rows(args.results, {int(n) for n in args.open.split()})
            if args.pr is not None:
                remove_pr_rows(args.results, args.pr)
        case "auto-merge":
            files = args.files.read_text().split()
            reason = auto_merge_reason(args.author, files)
            print("yes" if reason is None else f"no: {reason}")
        case "build":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(all_rows(args.results), indent=2) + "\n")


if __name__ == "__main__":
    main()
