"""Validation evidence shared by the runners and the documentation updater.

A "passed" record is only meaningful for exactly what produced it, so it carries
SHA-256 digests of the package source, the configuration inputs and (for the
notebook) every result product, plus the location of the ``aophoton`` module
that actually ran and a run identity.  Consumers re-check all of them against
the current tree before trusting the record.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

NOTEBOOK = "notebooks/eso_astrophotonics_showcase.ipynb"
RECORD = "results/notebook_execution.json"
JUNIT = "results/pytest.xml"
TEST_RECORD = "results/pytest_record.json"
# The implementation a run must have imported (relative to the project root).
IMPORT_TARGET = "src/aophoton/__init__.py"
# Files the notebook writes and the documentation updater (or a reader) consumes.
PRODUCT_PATTERNS = (
    "results/showcase_metrics.json", "results/design_map.npz", "results/port_trade.json",
    "results/parameter_provenance.json", "results/showcase_baseline/*",
    "figures/*.png", "figures/*.pdf",
)
# The complete set one notebook run must itself produce before it is certified.
FIGURE_NAMES = (
    "core_design_map", "coupling_sensitivity", "field_overlap", "rv_port_trade", "rv_recovery",
    "synthetic_observation", "system_overview", "wavefront_psf", "wavelength_seeing",
)
REQUIRED_PRODUCTS = (
    "results/showcase_metrics.json", "results/design_map.npz", "results/port_trade.json",
    "results/parameter_provenance.json", "results/showcase_baseline/summary.json",
    "results/showcase_baseline/spectrum.csv", "results/showcase_baseline/arrays.npz",
    *(f"figures/{name}.{suffix}" for name in FIGURE_NAMES for suffix in ("png", "pdf")),
)
# Environment variable through which tests/conftest.py reports the imported module.
IMPORT_RECORD_ENV = "AOPHOTON_IMPORT_RECORD"


def file_digest(path: Path) -> str:
    """SHA-256 of one file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


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
    root = Path(root)
    return _tree_digest(root, [*(root / "tests").rglob("*.py"), *(root / "scripts").glob("*.py")])


def inputs_digest(root: Path) -> str:
    """SHA-256 over the configuration inputs (``configs/``) the runs read."""
    root = Path(root)
    return _tree_digest(root, [p for p in (root / "configs").rglob("*") if p.is_file()])


def product_digests(root: Path) -> dict[str, str]:
    """``{relative path: SHA-256}`` of every current notebook result product."""
    root = Path(root)
    files = {p for pattern in PRODUCT_PATTERNS for p in root.glob(pattern) if p.is_file()}
    return {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(files)}


def local_import_env(root: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment whose ``PYTHONPATH`` puts this project's ``src`` first, so a
    run imports the code being certified rather than another installation."""
    env = dict(os.environ if base is None else base)
    src = str((Path(root) / "src").resolve())
    env["PYTHONPATH"] = os.pathsep.join([src, *filter(None, [env.get("PYTHONPATH")])])
    return env


def import_location(root: Path, module_file: str | Path) -> str:
    """Project-relative location of an imported ``aophoton`` (absolute if outside)."""
    module = Path(module_file).resolve()
    try:
        return module.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(module)


def _check_common(record: dict, root: Path, rerun: str) -> None:
    if record.get("source_sha256") != source_digest(root):
        raise RuntimeError(f"Package source changed since the recorded run; rerun {rerun}.")
    if record.get("inputs_sha256") != inputs_digest(root):
        raise RuntimeError(f"Configuration inputs changed since the recorded run; rerun {rerun}.")
    if record.get("imported_from") != IMPORT_TARGET:
        raise RuntimeError(
            f"The recorded run imported aophoton from {record.get('imported_from')!r}, not this "
            f"project's {IMPORT_TARGET}; rerun {rerun}.")


def verified_test_record(root: Path) -> dict:
    """Return the pytest record only if it passed and still matches the current tree.

    Raises ``RuntimeError`` for a failed run; when the package source, the
    configuration inputs, the tests/scripts or ``pytest.xml`` changed after the
    recorded run; or when that run imported ``aophoton`` from elsewhere.
    """
    root = Path(root)
    path = root / TEST_RECORD
    if not path.exists():
        raise RuntimeError("No pytest record; run scripts/run_tests.py.")
    record = json.loads(path.read_text())
    if record.get("status") != "passed":
        raise RuntimeError(f"Last test run did not pass (status={record.get('status')!r}).")
    _check_common(record, root, "scripts/run_tests.py")
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

    Raises ``RuntimeError`` when the last run did not pass; when the notebook,
    package source, configuration inputs or any result product changed after
    that run; when the run imported ``aophoton`` from elsewhere; or when the
    saved notebook itself does not show a clean execution.
    """
    import nbformat

    root = Path(root)
    record = json.loads((root / RECORD).read_text())
    if record.get("status") != "passed" or record.get("error_outputs") != 0:
        raise RuntimeError(f"Last notebook execution did not pass (status={record.get('status')!r}).")
    if record.get("notebook_sha256") != file_digest(root / NOTEBOOK):
        raise RuntimeError("Notebook changed since its recorded passing execution; rerun execute_showcase.py.")
    _check_common(record, root, "execute_showcase.py")
    missing = sorted(set(REQUIRED_PRODUCTS) - set(record.get("products_sha256") or {}))
    if missing:
        raise RuntimeError(f"The recorded notebook execution did not produce {missing}; rerun execute_showcase.py.")
    if record.get("products_sha256") != product_digests(root):
        raise RuntimeError("Result products (results/, figures/) changed since the recorded notebook "
                           "execution; rerun execute_showcase.py.")
    problems = notebook_problems(nbformat.read(root / NOTEBOOK, as_version=4))
    if problems:
        raise RuntimeError("Saved notebook is not a clean execution: " + "; ".join(problems))
    return record
