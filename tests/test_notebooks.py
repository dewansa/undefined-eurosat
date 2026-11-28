"""The worked examples' notebooks write their model.py: the two must not drift apart."""

import json
from pathlib import Path

import pytest

# (folder, the notebook whose submission cell writes model.py)
EXAMPLES = [
    (Path("submissions/demo"), "2-build-model.ipynb"),
    (Path("submissions/demo2"), "3-check-and-submit.ipynb"),
    (Path("submissions/autoresearch"), "3-confirm-and-submit.ipynb"),
]


@pytest.mark.parametrize(("folder", "notebook"), EXAMPLES, ids=[f.name for f, _ in EXAMPLES])
def test_the_submission_cell_matches_model_py(folder: Path, notebook: str) -> None:
    cells = json.loads((folder / notebook).read_text())["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    written = [c for c in code if c.startswith("%%writefile model.py\n")]
    assert len(written) == 1, "expected one cell that writes model.py"
    cell = written[0].removeprefix("%%writefile model.py\n")
    assert cell.rstrip("\n") == (folder / "model.py").read_text().rstrip("\n"), (
        "model.py differs from the notebook's submission cell: run that cell again"
    )
