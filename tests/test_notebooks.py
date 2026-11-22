"""The demo's notebook writes its model.py: the two must not drift apart."""

import json
from pathlib import Path

DEMO = Path("submissions/demo")


def test_the_submission_cell_matches_model_py() -> None:
    notebook = json.loads((DEMO / "2-build-model.ipynb").read_text())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    written = [c for c in cells if c.startswith("%%writefile model.py\n")]
    assert len(written) == 1, "expected one cell that writes model.py"
    cell = written[0].removeprefix("%%writefile model.py\n")
    assert cell.rstrip("\n") == (DEMO / "model.py").read_text().rstrip("\n"), (
        "model.py differs from the notebook's submission cell: run that cell again"
    )
