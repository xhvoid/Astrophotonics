"""Run the test suite and record evidence bound to the code and collection it tested."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import (  # noqa: E402
    IMPORT_TARGET, JUNIT, PYTEST_CONFIG, PYTEST_SELECTION_ENV, SESSION_RECORD_ENV, TEST_RECORD, atomic_write,
    collection_digest, file_digest, import_location, inputs_digest, local_import_env, source_digest,
    tests_digest, write_json,
)

# Serialises the test runs of one project (see ``run_lock``).  Not evidence;
# ignored by Git.
RUN_LOCK = "results/.pytest_run.lock"


@contextlib.contextmanager
def run_lock(root: Path):
    """Hold the project's exclusive test-run lock (blocking).

    A POSIX ``flock`` on ``RUN_LOCK``: it belongs to an open file description,
    so threads exclude each other as processes do, and the kernel releases it
    if its holder dies.  Each project has its own lock, so runs of different
    checkouts never wait for each other.
    """
    path = Path(root) / RUN_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o666)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for another test run of {root} to finish...", file=sys.stderr, flush=True)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)            # releases the lock


def _junit_counts(report: bytes) -> dict[str, int]:
    """The suite totals of a pytest JUnit report, or ``{}`` if it holds none."""
    try:
        suite = ET.fromstring(report).find("testsuite")
        return {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    except (ET.ParseError, AttributeError, KeyError, ValueError):
        return {}


def _session(root: Path, arguments: list[str], env: dict[str, str], report: Path,
             quiet: bool = False) -> tuple[int, dict]:
    """Run pytest under the project configuration; return its exit code and session report.

    A ``quiet`` session's console output is shown only if it fails.
    """
    command = [sys.executable, "-m", "pytest", "-c", str(root / PYTEST_CONFIG), "--rootdir", str(root), *arguments]
    result = subprocess.run(command, cwd=root, env={**env, SESSION_RECORD_ENV: str(report)},
                            capture_output=quiet, text=True)
    if quiet and result.returncode:
        print(result.stdout, result.stderr, sep="\n")
    return result.returncode, (json.loads(report.read_text()) if report.exists() else {})


def main(root: Path | None = None) -> int:
    """Collect the intended suite, run pytest with a JUnit report, then bind the outcome.

    The previous record is invalidated before the run, so a failed or
    interrupted run can never leave an older passing record in place.  The
    intended collection is what the project configuration (``pyproject.toml``)
    alone selects: it is collected with any caller options removed
    (``PYTEST_ADDOPTS``, ``PYTEST_PLUGINS``, ``addopts``).  The tests then run as
    the caller configured them.  The record passes only if exactly the intended
    tests ran and every one passed (a selection that passes is recorded as
    ``partial`` and certifies nothing), the session imported this project's
    source (``tests/conftest.py``), and nothing changed during the run.  It is
    bound to the source, configuration inputs, tests/scripts, pytest
    configuration, collection and JUnit file.  The test count is whatever the
    configuration collects.

    The record and ``pytest.xml`` are never written in place: pytest writes its
    JUnit report into this run's private scratch directory, and each file is
    published by ``atomic_write``, so a symbolic link at either path, even one
    made while the tests run, is replaced rather than written through.

    The JUnit counts and digest are taken from the bytes of that private
    report, read once, and exactly those bytes are published, so a record
    describes the report its own session wrote; the public ``pytest.xml`` is
    never read back.  A run whose session left no report (or one without suite
    totals) cannot pass, whatever file is at the public path.  One run of a
    project holds ``run_lock`` from before its ``running`` record and the
    removal of the old report until its final record is written (or an error
    leaves ``running`` in place), so a second runner of the same project waits,
    and says so, until the first has finished, then starts from its own
    ``running`` record.  The lock spans the whole run, not only the two final
    writes, so that no other run's ``running`` record or report removal can fall
    between them and no earlier run's final record can replace a later run's
    ``running`` one.
    """
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    record_path = root / TEST_RECORD
    junit = root / JUNIT
    with run_lock(root):
        run = dict(run_id=uuid.uuid4().hex, started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   source_sha256=source_digest(root), inputs_sha256=inputs_digest(root),
                   tests_sha256=tests_digest(root), pytest_config_sha256=file_digest(root / PYTEST_CONFIG))
        write_json(record_path, dict(status="running", **run))
        junit.unlink(missing_ok=True)       # a crashed run must not inherit old counts
        with tempfile.TemporaryDirectory(prefix="aophoton-tests-") as scratch:
            scratch = Path(scratch)
            clean = {key: value for key, value in os.environ.items() if key not in PYTEST_SELECTION_ENV}
            collect_code, intended = _session(
                root, ["--collect-only", "-q", "-p", "no:cacheprovider", "-o", "addopts="],
                local_import_env(root, clean), scratch / "intended.json", quiet=True)
            private = scratch / "pytest.xml"
            exit_code, session = _session(root, [f"--junitxml={private}", "-q"], local_import_env(root),
                                          scratch / "session.json")
            report = private.read_bytes() if private.is_file() else None
        if report is not None:
            atomic_write(junit, lambda stream: stream.write(report))
        intended_ids = set(intended.get("selected", []))
        selected = set(session.get("selected", []))
        outcomes = session.get("outcomes", {})
        imported = import_location(root, session["imported_from"]) if session.get("imported_from") else None
        record = dict(status="failed", **run, exit_code=exit_code, imported_from=imported,
                      python=platform.python_version(), pytest_addopts=os.environ.get("PYTEST_ADDOPTS"),
                      intended_tests=len(intended_ids), selected_tests=len(selected),
                      deselected=session.get("deselected", 0),
                      not_passed=sorted(n for n in selected if outcomes.get(n) != "passed"))
        if collect_code == 0 and intended_ids:
            record["collection_sha256"] = collection_digest(intended_ids)
        if report is not None:
            counts = _junit_counts(report)
            record.update(counts, junit_sha256=hashlib.sha256(report).hexdigest())
            clean_run = (exit_code == 0 and counts and counts["failures"] == counts["errors"] == counts["skipped"] == 0
                         and counts["tests"] == len(selected) and selected and not record["not_passed"])
            unchanged = (source_digest(root) == run["source_sha256"] and inputs_digest(root) == run["inputs_sha256"]
                         and tests_digest(root) == run["tests_sha256"]
                         and file_digest(root / PYTEST_CONFIG) == run["pytest_config_sha256"])
            if clean_run and unchanged and imported == IMPORT_TARGET and "collection_sha256" in record:
                record["status"] = "passed" if selected == intended_ids else "partial"
        write_json(record_path, record)
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
