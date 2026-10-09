"""Notebook-execution evidence shared by the runner and the documentation updater.

A "passed" record is only meaningful for the exact notebook and package source it
was produced from, so it carries their SHA-256 digests and a run identity, and
consumers re-check both before trusting it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

NOTEBOOK = "notebooks/eso_astrophotonics_showcase.ipynb"
RECORD = "results/notebook_execution.json"
JUNIT = "results/pytest.xml"
TEST_RECORD = "results/pytest_record.json"


def file_digest(path: Path) -> str:
    """SHA-256 of one file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(root: Path) -> str:
    """SHA-256 over every package source file, keyed by its relative path."""
    digest = hashlib.sha256()
    package = Path(root) / "src/aophoton"
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def tests_digest(root: Path) -> str:
    """SHA-256 over the test suite and the evidence scripts it exercises."""
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted([*(root / "tests").rglob("*.py"), *(root / "scripts").glob("*.py")]):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def verified_test_record(root: Path) -> dict:
    """Return the pytest record only if it passed and still matches the current tree.

    Raises ``RuntimeError`` for a failed run or when the package source, the
    tests/scripts, or ``pytest.xml`` changed after the recorded run.
    """
    root = Path(root)
    path = root / TEST_RECORD
    if not path.exists():
        raise RuntimeError("No pytest record; run scripts/run_tests.py.")
    record = json.loads(path.read_text())
    if record.get("status") != "passed":
        raise RuntimeError(f"Last test run did not pass (status={record.get('status')!r}).")
    if record.get("source_sha256") != source_digest(root):
        raise RuntimeError("Package source changed since the recorded test run; rerun scripts/run_tests.py.")
    if record.get("tests_sha256") != tests_digest(root):
        raise RuntimeError("Tests or scripts changed since the recorded test run; rerun scripts/run_tests.py.")
    if not (root / JUNIT).exists() or record.get("junit_sha256") != file_digest(root / JUNIT):
        raise RuntimeError("pytest.xml does not match the recorded test run; rerun scripts/run_tests.py.")
    return record


def notebook_problems(notebook: Any) -> list[str]:
    """Reasons an executed notebook object does not demonstrate a clean run."""
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    problems = []
    if not code_cells:
        problems.append("notebook has no code cells")
    if any(cell.execution_count is None for cell in code_cells):
        problems.append("not every code cell was executed")
    if any(output.output_type == "error" for cell in code_cells for output in cell.outputs):
        problems.append("notebook contains error outputs")
    if not all(cell.outputs for cell in code_cells):
        problems.append("a code cell has no displayed result")
    return problems


def write_record(root: Path, record: dict) -> None:
    """Replace the execution record (the previous evidence is always discarded)."""
    (Path(root) / RECORD).write_text(json.dumps(record, indent=2) + "\n")


def verified_record(root: Path) -> dict:
    """Return the execution record only if it still describes the current files.

    Raises ``RuntimeError`` when the last run did not pass, or when the notebook
    or package source changed after that run, or when the saved notebook itself
    does not show a clean execution.
    """
    import nbformat

    root = Path(root)
    record = json.loads((root / RECORD).read_text())
    if record.get("status") != "passed" or record.get("error_outputs") != 0:
        raise RuntimeError(f"Last notebook execution did not pass (status={record.get('status')!r}).")
    if record.get("notebook_sha256") != file_digest(root / NOTEBOOK):
        raise RuntimeError("Notebook changed since its recorded passing execution; rerun execute_showcase.py.")
    if record.get("source_sha256") != source_digest(root):
        raise RuntimeError("Package source changed since the recorded notebook execution; rerun execute_showcase.py.")
    problems = notebook_problems(nbformat.read(root / NOTEBOOK, as_version=4))
    if problems:
        raise RuntimeError("Saved notebook is not a clean execution: " + "; ".join(problems))
    return record
