"""Execute the saved showcase in this Python environment and verify every output."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import (  # noqa: E402
    FIGURE_NAMES, IMPORT_TARGET, NOTEBOOK, REQUIRED_PRODUCTS, file_digest, import_location, inputs_digest,
    local_import_env, notebook_problems, product_digests, source_digest, write_record,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _kernel_import_path(kernel: KernelManager) -> str:
    """Ask the still-running kernel which aophoton module the notebook used."""
    client = kernel.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        lines: list[str] = []

        def collect(message: dict) -> None:
            if message["msg_type"] == "stream":
                lines.append(message["content"]["text"])

        client.execute_interactive("import aophoton, pathlib; print(pathlib.Path(aophoton.__file__).resolve())",
                                   output_hook=collect, timeout=60)
        return "".join(lines).strip()
    finally:
        client.stop_channels()


def _stage_inputs(root: Path, stage: Path) -> None:
    """Fresh kernel working directory holding only what the notebook reads.

    The notebook resolves its project root from the kernel's working directory,
    so its results/ and figures/ start empty here: every product found after the
    run was written by this run.  ``configs/`` is LINKED, not copied: the
    configuration loader resolves the link, so template paths given relative to
    a YAML file resolve against its real location, wherever they point.
    """
    (stage / "configs").symlink_to((root / "configs").resolve(), target_is_directory=True)


def _writable_location(path: Path) -> bool:
    """Whether ``path`` can be replaced or removed (or created in its nearest existing ancestor)."""
    if path.exists() and not os.access(path, os.W_OK):
        return False
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    return os.access(parent, os.W_OK)


def _replace_file(source: Path, target: Path) -> None:
    """Atomically put a copy of ``source`` at ``target`` (temporary file + rename)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.publish-tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _restore(root: Path, backup: Path, previous: dict[str, str]) -> None:
    """Return the published products to exactly ``previous`` (from ``backup``)."""
    for relative in sorted(set(product_digests(root)) - set(previous)):
        (root / relative).unlink()
    for relative, digest in previous.items():
        target = root / relative
        if not target.exists() or file_digest(target) != digest:
            _replace_file(backup / relative, target)
    if product_digests(root) != previous:
        raise RuntimeError(f"Rollback incomplete; the previous products are preserved in {backup}.")


def _publish(stage: Path, root: Path, produced: dict[str, str]) -> None:
    """Make the project's product set exactly the staged one, all or nothing.

    A destination that cannot be written is refused before anything changes.
    The previously published set is backed up; produced files then replace their
    published copies one atomic rename at a time, and only once all of them are
    in place are stale products of earlier runs removed.  Any failure restores
    the previous set exactly (verified by digest) before the error propagates,
    so a failed publication never leaves old and new products mixed.  If the
    restore itself fails, the complete backup is kept and the error names it.
    """
    previous = product_digests(root)
    stale = sorted(set(previous) - set(produced))
    blocked = [relative for relative in sorted(set(produced) | set(stale))
               if not _writable_location(root / relative)]
    if blocked:
        raise PermissionError(f"Cannot publish: destinations are not writable {blocked}; nothing was changed.")
    backup = Path(tempfile.mkdtemp(prefix=".publish-backup-", dir=root))
    restored = True
    try:
        for relative in previous:
            (backup / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, backup / relative)
        try:
            for relative in produced:
                _replace_file(stage / relative, root / relative)
            for relative in stale:
                (root / relative).unlink()
            if product_digests(root) != produced:
                raise RuntimeError("Published products differ from the staged run.")
        except BaseException as error:
            restored = False
            try:
                _restore(root, backup, previous)
            except BaseException as restore_error:
                # The backup is retained (restored stays False): say where it is.
                raise RuntimeError(
                    f"Publication failed ({type(error).__name__}: {error}) and restoring the previous "
                    f"products also failed ({type(restore_error).__name__}: {restore_error}); the complete "
                    f"previous product set is preserved in {backup}.") from restore_error
            restored = True
            raise
    finally:
        if restored:
            shutil.rmtree(backup, ignore_errors=True)


def main(root: Path | None = None) -> None:
    """Execute a fresh kernel without installing a machine-global kernelspec.

    The previous execution record is invalidated before the run starts, a
    failure is recorded as such, and a passing record is bound to the digests of
    the notebook, package source, configuration inputs and every result product,
    and to the aophoton module the kernel actually imported (the kernel runs with
    this project's ``src`` first on ``PYTHONPATH``; any other module fails the run).

    The notebook runs in a fresh staging directory (``_stage_inputs``), so its
    products are certified only if this run wrote every file in
    ``REQUIRED_PRODUCTS``; a skipped or removed exporter fails the run instead of
    certifying an older file.  Products are published into ``results/`` and
    ``figures/`` only after every check has passed; a failed run leaves the
    previously published products untouched.
    """
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".jupyter_runtime"))
    path = root / NOTEBOOK
    run = dict(run_id=uuid.uuid4().hex, started_utc=_now(), source_sha256=source_digest(root),
               inputs_sha256=inputs_digest(root))
    write_record(root, dict(status="running", **run))
    start = time.perf_counter()
    try:
        notebook = nbformat.read(path, as_version=4)
        with tempfile.TemporaryDirectory(prefix="aophoton-showcase-") as scratch:
            stage = Path(scratch)
            _stage_inputs(root, stage)
            kernel = KernelManager()
            # Keep execution on the exact interpreter used to launch this script.
            kernel._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                             display_name="aophoton validation", language="python")
            client = NotebookClient(notebook, km=kernel, timeout=240, allow_errors=False,
                                    resources={"metadata": {"path": str(stage)}})
            try:
                client.execute(env=local_import_env(root))
                imported = import_location(root, _kernel_import_path(kernel))
            finally:
                # Persist partial outputs for diagnosis if execution fails; a successful
                # delivery additionally requires all assertions below and exit status zero.
                nbformat.write(notebook, path)
                # An externally supplied manager belongs to this script; nbclient does
                # not terminate it automatically. Close it even if a cell raises.
                if kernel.has_kernel:
                    kernel.shutdown_kernel(now=True)
            problems = notebook_problems(notebook)
            assert not problems, "; ".join(problems)
            if imported != IMPORT_TARGET:
                raise RuntimeError(f"Notebook kernel imported aophoton from {imported!r}, not {IMPORT_TARGET}.")
            produced = product_digests(stage)
            missing = sorted(set(REQUIRED_PRODUCTS) - set(produced))
            if missing:
                raise RuntimeError(f"Notebook run did not produce required outputs {missing}.")
            assert all((stage / f"figures/{name}.pdf").stat().st_size > 0 for name in FIGURE_NAMES)
            if source_digest(root) != run["source_sha256"] or inputs_digest(root) != run["inputs_sha256"]:
                raise RuntimeError("Package source or configuration inputs changed during notebook execution.")
            _publish(stage, root, produced)
    except BaseException as error:
        write_record(root, dict(status="failed", **run, finished_utc=_now(),
                                error=f"{type(error).__name__}: {error}",
                                wall_time_s=time.perf_counter()-start))
        raise
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    report = dict(status="passed", **run, finished_utc=_now(), notebook_sha256=file_digest(path),
                  imported_from=imported, products_sha256=produced,
                  code_cells=len(code_cells), error_outputs=0,
                  png_figures=len(FIGURE_NAMES), pdf_figures=len(FIGURE_NAMES),
                  wall_time_s=time.perf_counter()-start, python=sys.version,
                  executable=sys.executable)
    write_record(root, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
