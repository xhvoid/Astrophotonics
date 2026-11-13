"""Execute the saved showcase in this Python environment and verify every output."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import uuid

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import (  # noqa: E402
    FIGURE_NAMES, IMPORT_TARGET, NOTEBOOK, OWNERSHIP, RECORD, REQUIRED_PRODUCTS, atomic_write, file_digest,
    import_location, inputs_digest, is_product_path, local_import_env, notebook_problems, owned_products,
    product_digests, product_paths, source_digest, write_record,
)

# Serialises everything that reads or writes a project's published state (see
# ``publication_lock``).  Not a product; ignored by Git.
PUBLICATION_LOCK = "results/.publication.lock"
# Where a failed run keeps its notebook, as ``<run_id>.ipynb``.  Not a product
# location, so nothing here is published, certified or retired; ignored by Git.
FAILED_EXECUTIONS = "results/failed_executions"
_held = threading.local()


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


def _report_wait(root: Path) -> None:
    print(f"Waiting for another publication into {root} to finish...", file=sys.stderr, flush=True)


@contextlib.contextmanager
def publication_lock(root: Path):
    """Hold the project's exclusive publication lock (blocking, re-entrant per thread).

    The ownership snapshot, product and notebook replacement, manifest,
    execution record, verification and rollback of one publication all happen
    under it, so independent processes publishing into one project commit
    whole generations one after another and a failing publisher can only ever
    restore the generation that was committed when it took the lock.  It is a
    POSIX ``flock`` on ``PUBLICATION_LOCK``: the lock belongs to an open file
    description, so threads of one process exclude each other as processes do,
    and the kernel releases it if its holder dies.
    """
    key = str(Path(root).resolve())
    depth = _held.__dict__.setdefault("depth", {})
    if depth.get(key):
        depth[key] += 1
        try:
            yield
        finally:
            depth[key] -= 1
        return
    path = Path(root) / PUBLICATION_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o666)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _report_wait(root)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        depth[key] = 1
        try:
            yield
        finally:
            depth[key] = 0
    finally:
        os.close(descriptor)            # releases the lock


def _writable_location(path: Path) -> bool:
    """Whether ``path`` can be replaced or removed (or created in its nearest existing ancestor)."""
    if path.exists() and not os.access(path, os.W_OK):
        return False
    parent = path.parent
    while not parent.exists():
        parent = parent.parent
    return os.access(parent, os.W_OK)


def _replace_file(source: Path, target: Path) -> None:
    """Atomically put a copy of ``source`` (bytes, mode and times) at ``target``."""
    with open(source, "rb") as original:
        atomic_write(target, lambda stream: shutil.copyfileobj(original, stream), like=source)


def _write_ownership(root: Path, produced: dict[str, str], run_id: str | None) -> None:
    """Atomically record ``produced`` as the products notebook runs now own."""
    text = json.dumps(dict(run_id=run_id, products_sha256=produced), indent=2) + "\n"
    atomic_write(root / OWNERSHIP, lambda stream: stream.write(text.encode()))


def _digests(root: Path, paths) -> dict[str, str]:
    return {relative: file_digest(root / relative) for relative in paths if (root / relative).is_file()}


def _identities(root: Path, exclude) -> dict[str, tuple[int, int, int]]:
    """Files in the product locations outside ``exclude``, with their inode, size
    and modification time: enough to see a change without reading any content."""
    return {relative: (status.st_ino, status.st_size, status.st_mtime_ns)
            for relative in product_paths(root) if relative not in exclude
            for status in [(root / relative).lstat()]}


def _restore(root: Path, backup: Path, touched: list[str], previous: dict[str, str]) -> None:
    """Return every path the publication touched to ``previous`` (from ``backup``)."""
    for relative in touched:
        target = root / relative
        if relative not in previous:
            target.unlink(missing_ok=True)
        elif not target.is_file() or file_digest(target) != previous[relative]:
            _replace_file(backup / relative, target)
    if _digests(root, touched) != previous:
        raise RuntimeError(f"Rollback incomplete; the previous products are preserved in {backup}.")


def _publish(stage: Path, root: Path, produced: dict[str, str], owned: dict[str, str] | None = None,
             run_id: str | None = None, companions: tuple[str, ...] = (), record: dict | None = None) -> None:
    """Publish the staged products, all or nothing, touching only what notebook runs own.

    Runs under ``publication_lock``.  ``owned`` is ownership as it stood
    before this run (``owned_products``); the current manifest, if there is
    one, takes precedence, because another publisher may have committed since
    the run started.  Produced files and ``companions`` (staged files published
    and rolled back with the products but not owned as products: the executed
    notebook) replace their published copies one atomic rename at a time.  Only
    once all are in place is a previously owned product that this run no longer
    writes retired, and only if it is still byte-identical to what the notebook
    wrote; an edited or unrecorded file has ambiguous ownership and is kept.
    Any other file in results/ or figures/ (a user's scan, notes) is listed but
    never read, backed up, replaced or removed.  The ownership manifest then
    lists exactly the produced set.

    ``record`` (the passing execution record) is the commit: it is written
    last, by one atomic rename, after the published set has been verified, and
    the backup of the previous generation is kept until it has succeeded.  A
    failure up to and including that rename therefore restores the previous
    generation and leaves the previous record file unchanged.

    A destination that cannot be written, or a produced path outside the
    declared products that would overwrite a file no run owns, is refused before
    anything changes.  Every touched path is backed up; any failure restores
    them exactly (verified by digest) before the error propagates.  If the
    restore itself fails, the backup is kept and the error names it.
    """
    with publication_lock(root):
        if (root / OWNERSHIP).exists() or owned is None:
            owned = owned_products(root)
        retired = sorted(relative for relative, digest in owned.items()
                         if relative not in produced and is_product_path(relative)
                         and (root / relative).is_file() and file_digest(root / relative) == digest)
        foreign = sorted(relative for relative in produced if relative not in REQUIRED_PRODUCTS
                         and (root / relative).is_file() and owned.get(relative) != file_digest(root / relative))
        if foreign:
            raise FileExistsError(f"Cannot publish: {foreign} would overwrite files no notebook run wrote; "
                                  "nothing was changed.")
        staged = dict(produced, **_digests(stage, companions))
        touched = sorted(set(staged) | set(retired) | {OWNERSHIP})
        committed = [RECORD] if record is not None else []
        blocked = [relative for relative in touched + committed if not _writable_location(root / relative)]
        if blocked:
            raise PermissionError(f"Cannot publish: destinations are not writable {blocked}; nothing was changed.")
        previous = _digests(root, touched)
        others = _identities(root, touched)
        backup = Path(tempfile.mkdtemp(prefix=".publish-backup-", dir=root))
        restored = True
        try:
            for relative in previous:
                (backup / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / relative, backup / relative)
            try:
                for relative in staged:
                    _replace_file(stage / relative, root / relative)
                for relative in retired:
                    (root / relative).unlink()
                _write_ownership(root, produced, run_id)
                if _digests(root, staged) != staged or any((root / r).exists() for r in retired) \
                        or _identities(root, touched) != others:
                    raise RuntimeError("Published products differ from the staged run.")
                if record is not None:
                    write_record(root, record)          # the commit; nothing can fail after it
            except BaseException as error:
                restored = False
                try:
                    _restore(root, backup, touched, previous)
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


def _save_notebook(notebook, path: Path) -> None:
    """Atomically write ``notebook`` (the bytes ``nbformat.write`` produces)."""
    text = nbformat.writes(notebook)
    text += "" if text.endswith("\n") else "\n"
    atomic_write(path, lambda stream: stream.write(text.encode()))


def _record_failure(root: Path, run: dict, start: float, error: BaseException, notebook=None) -> None:
    """Record this run as the latest attempt, failed, under ``publication_lock``.

    ``notebook`` (the run's notebook as far as it executed) is kept for
    diagnosis at ``FAILED_EXECUTIONS/<run_id>.ipynb`` and named in the record;
    the published notebook belongs to the committed generation, which may be
    another run's, and is never touched.  If the notebook cannot be saved, the
    record says why.  If the record itself cannot be written, the previous
    record stays in place (a running or another run's record) and the reason is
    printed; the caller still raises ``error``.
    """
    failure = dict(status="failed", **run, finished_utc=_now(), error=f"{type(error).__name__}: {error}",
                   wall_time_s=time.perf_counter()-start)
    if notebook is not None:
        diagnostic = f"{FAILED_EXECUTIONS}/{run['run_id']}.ipynb"
        try:
            _save_notebook(notebook, root / diagnostic)
            failure["diagnostic_notebook"] = diagnostic
        except OSError as save_error:
            failure["diagnostic_notebook_error"] = f"{type(save_error).__name__}: {save_error}"
    try:
        write_record(root, failure)
    except OSError as record_error:
        print(f"Could not record this run's failure in {root / RECORD} "
              f"({type(record_error).__name__}: {record_error}).", file=sys.stderr, flush=True)


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
    certifying an older file.  The executed notebook, its products, the
    ownership manifest and, last, the passing record are then published as one
    generation under ``publication_lock``, only after every check has passed
    (``_publish``): a failure at any step up to the record's commit restores the
    generation committed when the lock was taken, which may be another
    publisher's.  The lock is held from then until a failure is recorded, so no
    other run commits in between.  A failed execution never publishes and never
    touches the published notebook; its notebook is kept under its run ID
    (``_record_failure``).  Ownership is read before the running record
    replaces the previous one, so a legacy passing record can still identify
    the products it wrote (``owned_products``).
    """
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".jupyter_runtime"))
    path = root / NOTEBOOK
    run = dict(run_id=uuid.uuid4().hex, started_utc=_now(), source_sha256=source_digest(root),
               inputs_sha256=inputs_digest(root))
    with publication_lock(root):
        owned = owned_products(root)
        write_record(root, dict(status="running", **run))
    start = time.perf_counter()
    executed = None             # the notebook once execution starts: kept if the run fails
    with contextlib.ExitStack() as held:
        try:
            notebook = nbformat.read(path, as_version=4)
            with tempfile.TemporaryDirectory(prefix="aophoton-showcase-") as scratch:
                stage = Path(scratch)
                _stage_inputs(root, stage)
                kernel = KernelManager()
                # Keep execution on the exact interpreter used to launch this script.
                kernel._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f",
                                                       "{connection_file}"],
                                                 display_name="aophoton validation", language="python")
                client = NotebookClient(notebook, km=kernel, timeout=240, allow_errors=False,
                                        resources={"metadata": {"path": str(stage)}})
                try:
                    executed = notebook
                    client.execute(env=local_import_env(root))
                    imported = import_location(root, _kernel_import_path(kernel))
                finally:
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
                _save_notebook(notebook, stage / NOTEBOOK)
                code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
                held.enter_context(publication_lock(root))
                report = dict(status="passed", **run, finished_utc=_now(),
                              notebook_sha256=file_digest(stage / NOTEBOOK), imported_from=imported,
                              products_sha256=produced, code_cells=len(code_cells), error_outputs=0,
                              png_figures=len(FIGURE_NAMES), pdf_figures=len(FIGURE_NAMES),
                              wall_time_s=time.perf_counter()-start, python=sys.version,
                              executable=sys.executable)
                _publish(stage, root, produced, owned, run["run_id"], companions=(NOTEBOOK,), record=report)
        except BaseException as error:
            held.enter_context(publication_lock(root))          # re-entrant if publication holds it
            _record_failure(root, run, start, error, executed)
            raise
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
