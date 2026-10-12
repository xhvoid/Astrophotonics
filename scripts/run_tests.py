"""Run the test suite and record evidence bound to the code it tested."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import (  # noqa: E402
    IMPORT_RECORD_ENV, IMPORT_TARGET, JUNIT, TEST_RECORD, file_digest, import_location, inputs_digest,
    local_import_env, source_digest, tests_digest,
)


def main(root: Path | None = None) -> int:
    """Run pytest with a JUnit report, then bind the outcome to what was tested.

    The previous record is invalidated before the run, so a failed or
    interrupted run can never leave an older passing record in place.  Tests run
    with this project's ``src`` first on ``PYTHONPATH``, and the session reports
    the ``aophoton`` module it imported (``tests/conftest.py``); the record passes
    only if that is this project's source, and it is bound to the source, the
    configuration inputs, the tests/scripts and the JUnit file.
    """
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    record_path = root / TEST_RECORD
    run = dict(run_id=uuid.uuid4().hex, started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               source_sha256=source_digest(root), inputs_sha256=inputs_digest(root),
               tests_sha256=tests_digest(root))
    record_path.write_text(json.dumps(dict(status="running", **run), indent=2) + "\n")
    junit = root / JUNIT
    junit.unlink(missing_ok=True)       # a crashed run must not inherit old counts
    with tempfile.TemporaryDirectory(prefix="aophoton-tests-") as scratch:
        report = Path(scratch) / "imported_from.txt"
        env = local_import_env(root)
        env[IMPORT_RECORD_ENV] = str(report)
        exit_code = subprocess.call([sys.executable, "-m", "pytest", f"--junitxml={junit}", "-q"],
                                    cwd=root, env=env)
        imported = import_location(root, report.read_text().strip()) if report.exists() else None
    record = dict(status="failed", **run, exit_code=exit_code, imported_from=imported)
    if junit.exists():
        suite = ET.parse(junit).getroot().find("testsuite")
        counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")} if suite is not None else {}
        record.update(counts, junit_sha256=file_digest(junit))
        clean = counts and counts["failures"] == counts["errors"] == counts["skipped"] == 0
        unchanged = (source_digest(root) == run["source_sha256"] and inputs_digest(root) == run["inputs_sha256"]
                     and tests_digest(root) == run["tests_sha256"])
        if exit_code == 0 and clean and unchanged and imported == IMPORT_TARGET:
            record["status"] = "passed"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
