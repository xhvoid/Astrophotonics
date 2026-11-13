"""Validation evidence shared by the runners and the documentation updater.

A "passed" record is only meaningful for exactly what produced it, so it carries
SHA-256 digests of the package source, the configuration inputs and (for the
notebook) every result product, plus the location of the ``aophoton`` module
that actually ran and a run identity.  Consumers re-check all of them against
the current tree before trusting the record.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any

NOTEBOOK = "notebooks/eso_astrophotonics_showcase.ipynb"
RECORD = "results/notebook_execution.json"
JUNIT = "results/pytest.xml"
TEST_RECORD = "results/pytest_record.json"
# Written by every successful publication: the products notebook runs own and
# may therefore later replace or retire.  Other files under results/ and figures/
# (a user's own scan or notes) are not products and are never touched.
OWNERSHIP = "results/notebook_products.json"
# The implementation a run must have imported (relative to the project root).
IMPORT_TARGET = "src/aophoton/__init__.py"
# Where notebook products may live.  Matching a pattern does not make a file a
# product: ownership comes only from OWNERSHIP (see ``owned_products``).
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
# Environment variable through which tests/conftest.py reports the imported
# module, the selected test IDs and their outcomes.
SESSION_RECORD_ENV = "AOPHOTON_SESSION_RECORD"
# The pytest configuration that defines the intended suite (testpaths, options).
PYTEST_CONFIG = "pyproject.toml"
# Environment variables through which a caller can change which tests run.
PYTEST_SELECTION_ENV = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")
# Permission bits for new files written by ``atomic_write`` (mkstemp creates them 0600).
_UMASK = os.umask(0)
os.umask(_UMASK)


def atomic_write(target: Path, fill, like: Path | None = None) -> None:
    """Put what ``fill(stream)`` writes at ``target`` with one atomic rename.

    The temporary file is created exclusively (``mkstemp``: a fresh random name,
    opened with O_EXCL, so it never follows or reuses an existing file or link)
    in the destination directory, and only that file is removed if anything
    fails, leaving the previous ``target`` as it was.  The rename replaces the
    directory entry itself, so a symbolic link at ``target`` is replaced, not
    written through.  Mode and times are copied from ``like``, or the mode
    follows the umask.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".publish-tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            fill(stream)
        if like is None:
            os.chmod(temporary, 0o666 & ~_UMASK)
        else:
            shutil.copystat(like, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def collection_digest(node_ids) -> str:
    """SHA-256 over a set of pytest node IDs (order-independent)."""
    return hashlib.sha256("\n".join(sorted(node_ids)).encode()).hexdigest()


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


def _referenced_templates(config_file: Path) -> list[Path]:
    """Template files a configuration names (``observation.template_path``),
    resolved the way the configuration loader resolves them: relative to the YAML."""
    import yaml

    try:
        raw = yaml.safe_load(config_file.read_text())
    except (OSError, yaml.YAMLError):
        return []
    observation = raw.get("observation") if isinstance(raw, dict) else None
    template = observation.get("template_path") if isinstance(observation, dict) else None
    return [(config_file.resolve().parent / template).resolve()] if isinstance(template, str) else []


def inputs_digest(root: Path) -> str:
    """SHA-256 over the configuration inputs the runs read: every file in
    ``configs/`` and every template file those configurations reference."""
    root = Path(root)
    configs = sorted(p for p in (root / "configs").rglob("*") if p.is_file())
    digest = hashlib.sha256(_tree_digest(root, configs).encode())
    templates = {import_location(root, t): t for c in configs if c.suffix in {".yaml", ".yml"}
                 for t in _referenced_templates(c)}
    # Ordered by the portable labels that are hashed (project-relative inside,
    # absolute outside), so moving the project cannot reorder unchanged inputs.
    for label in sorted(templates):
        template = templates[label]
        digest.update(label.encode() + b"\0")
        digest.update((template.read_bytes() if template.is_file() else b"<missing>") + b"\0")
    return digest.hexdigest()


def product_paths(root: Path) -> list[str]:
    """Relative paths of every file in the product locations, whether or not a
    notebook run wrote it.  Lists names only: no file content is read, so an
    unrelated user file's size or read permission cannot affect the caller."""
    root = Path(root)
    files = {p for pattern in PRODUCT_PATTERNS for p in root.glob(pattern) if p.is_file()}
    return sorted(p.relative_to(root).as_posix() for p in files)


def product_digests(root: Path) -> dict[str, str]:
    """``{relative path: SHA-256}`` of every file in the product locations.
    Reads them all: meant for a staging directory a run filled itself."""
    return {relative: file_digest(Path(root) / relative) for relative in product_paths(root)}


def is_product_path(relative: str) -> bool:
    """Whether a project-relative POSIX path lies in a product location.

    Matched component by component, so ``figures/*.png`` covers neither
    ``figures/sub/x.png`` nor anything outside the project.
    """
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        return False
    return any(len(path.parts) == len(PurePosixPath(pattern).parts)
               and all(fnmatchcase(part, glob) for part, glob in zip(path.parts, PurePosixPath(pattern).parts))
               for pattern in PRODUCT_PATTERNS)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def owned_products(root: Path) -> dict[str, str]:
    """``{relative path: SHA-256 it was written with}`` for the products an
    earlier notebook publication demonstrably wrote.

    Read from the ownership manifest.  A tree published before manifests existed
    falls back to the products of its *passing* execution record, which that
    publisher made exactly the staged set.  An unreadable manifest, a legacy
    record that did not pass, or neither file means nothing is owned, so nothing
    will be retired.  Entries outside the product locations are ignored.
    """
    root = Path(root)
    if (root / OWNERSHIP).exists():
        data = _read_json(root / OWNERSHIP)
    else:
        data = _read_json(root / RECORD)
        if not isinstance(data, dict) or data.get("status") != "passed":
            data = None
    products = data.get("products_sha256") if isinstance(data, dict) else None
    if not isinstance(products, dict):
        return {}
    return {path: digest for path, digest in products.items()
            if isinstance(path, str) and isinstance(digest, str) and is_product_path(path)}


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
    """Return the pytest record only if it certifies the full suite of the current tree.

    Raises ``RuntimeError`` for a failed run; for a partial run (a selection of
    the intended collection, however small or large); for a record that does
    not bind its collection; when the package source, the configuration inputs,
    the tests/scripts, the pytest configuration or ``pytest.xml`` changed after
    the recorded run; or when that run imported ``aophoton`` from elsewhere.
    """
    root = Path(root)
    path = root / TEST_RECORD
    if not path.exists():
        raise RuntimeError("No pytest record; run scripts/run_tests.py.")
    record = json.loads(path.read_text())
    if record.get("status") == "partial":
        raise RuntimeError(
            f"Last test run was partial: it ran {record.get('tests')} tests where the project configuration "
            f"collects {record.get('intended_tests')} (PYTEST_ADDOPTS={record.get('pytest_addopts')!r}), so it "
            "cannot certify the full suite; rerun scripts/run_tests.py without test-selection options.")
    if record.get("status") != "passed":
        raise RuntimeError(f"Last test run did not pass (status={record.get('status')!r}).")
    _check_common(record, root, "scripts/run_tests.py")
    if not record.get("collection_sha256") or record.get("tests") != record.get("intended_tests"):
        raise RuntimeError("The pytest record does not bind the intended test collection; rerun scripts/run_tests.py.")
    if record.get("pytest_config_sha256") != file_digest(root / PYTEST_CONFIG):
        raise RuntimeError(f"{PYTEST_CONFIG} changed since the recorded test run; rerun scripts/run_tests.py.")
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


def write_json(path: Path, data: Any) -> None:
    """Replace ``path`` with ``data`` as indented JSON.

    Written by ``atomic_write``: a symbolic link at ``path`` is replaced by a
    regular file, never written through, and a failed write leaves the
    previous file in place.
    """
    text = json.dumps(data, indent=2) + "\n"
    atomic_write(Path(path), lambda stream: stream.write(text.encode()))


def write_record(root: Path, record: dict) -> None:
    """Replace the execution record (the previous evidence is always discarded) via ``write_json``."""
    write_json(Path(root) / RECORD, record)


def verified_record(root: Path) -> dict:
    """Return the execution record only if it still describes the current files.

    Raises ``RuntimeError`` when the last run did not pass; when the notebook,
    package source, configuration inputs or any product that run wrote changed
    (or disappeared) after it; when the run imported ``aophoton`` from
    elsewhere; or when the saved notebook itself does not show a clean execution.
    Other files in the product locations are not products of the run: they
    neither invalidate the record nor are vouched for by it (``unowned_files``).
    """
    import nbformat

    root = Path(root)
    record = json.loads((root / RECORD).read_text())
    if record.get("status") != "passed" or record.get("error_outputs") != 0:
        raise RuntimeError(f"Last notebook execution did not pass (status={record.get('status')!r}).")
    if record.get("notebook_sha256") != file_digest(root / NOTEBOOK):
        raise RuntimeError("Notebook changed since its recorded passing execution; rerun execute_showcase.py.")
    _check_common(record, root, "execute_showcase.py")
    recorded = record.get("products_sha256") or {}
    missing = sorted(set(REQUIRED_PRODUCTS) - set(recorded))
    if missing:
        raise RuntimeError(f"The recorded notebook execution did not produce {missing}; rerun execute_showcase.py.")
    current = {relative: file_digest(root / relative) for relative in recorded if (root / relative).is_file()}
    if current != recorded:
        raise RuntimeError("Result products (results/, figures/) changed since the recorded notebook "
                           "execution; rerun execute_showcase.py.")
    problems = notebook_problems(nbformat.read(root / NOTEBOOK, as_version=4))
    if problems:
        raise RuntimeError("Saved notebook is not a clean execution: " + "; ".join(problems))
    return record


def unowned_files(root: Path, record: dict) -> list[str]:
    """Files in the product locations that ``record``'s run did not write (names only, never read)."""
    return sorted(set(product_paths(root)) - set(record.get("products_sha256") or {}))
