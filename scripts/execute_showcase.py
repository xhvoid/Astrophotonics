"""Execute the saved showcase in this Python environment and verify every output."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import NOTEBOOK, file_digest, notebook_problems, source_digest, write_record  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(root: Path | None = None) -> None:
    """Execute a fresh kernel without installing a machine-global kernelspec.

    The previous execution record is invalidated before the run starts, a
    failure is recorded as such, and a passing record is bound to the digests of
    the notebook and package source it was produced from.
    """
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".ipython"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".jupyter_runtime"))
    path = root / NOTEBOOK
    run = dict(run_id=uuid.uuid4().hex, started_utc=_now(), source_sha256=source_digest(root))
    write_record(root, dict(status="running", **run))
    start = time.perf_counter()
    try:
        notebook = nbformat.read(path, as_version=4)
        kernel = KernelManager()
        # Keep execution on the exact interpreter used to launch this script.
        kernel._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                         display_name="aophoton validation", language="python")
        client = NotebookClient(notebook, km=kernel, timeout=240, allow_errors=False,
                                resources={"metadata": {"path": str(root)}})
        try:
            client.execute()
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
        figures = sorted((root / "figures").glob("*.png"))
        assert len(figures) >= 9
        assert all(p.with_suffix(".pdf").stat().st_size > 0 for p in figures)
        if source_digest(root) != run["source_sha256"]:
            raise RuntimeError("Package source changed during notebook execution.")
    except BaseException as error:
        write_record(root, dict(status="failed", **run, finished_utc=_now(),
                                error=f"{type(error).__name__}: {error}",
                                wall_time_s=time.perf_counter()-start))
        raise
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    report = dict(status="passed", **run, finished_utc=_now(), notebook_sha256=file_digest(path),
                  code_cells=len(code_cells), error_outputs=0,
                  png_figures=len(figures), pdf_figures=len(figures),
                  wall_time_s=time.perf_counter()-start, python=sys.version,
                  executable=sys.executable)
    write_record(root, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
