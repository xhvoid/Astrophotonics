"""Run the test suite and record evidence bound to the code it tested."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import JUNIT, TEST_RECORD, file_digest, source_digest, tests_digest  # noqa: E402


def main(root: Path | None = None) -> int:
    """Run pytest with a JUnit report, then bind the outcome to source/test digests.

    The previous record is invalidated before the run, so a failed or
    interrupted run can never leave an older passing record in place.
    """
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    record_path = root / TEST_RECORD
    run = dict(run_id=uuid.uuid4().hex, started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               source_sha256=source_digest(root), tests_sha256=tests_digest(root))
    record_path.write_text(json.dumps(dict(status="running", **run), indent=2) + "\n")
    junit = root / JUNIT
    junit.unlink(missing_ok=True)       # a crashed run must not inherit old counts
    exit_code = subprocess.call([sys.executable, "-m", "pytest", f"--junitxml={junit}", "-q"], cwd=root)
    record = dict(status="failed", **run, exit_code=exit_code)
    if junit.exists():
        suite = ET.parse(junit).getroot().find("testsuite")
        counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")} if suite is not None else {}
        record.update(counts, junit_sha256=file_digest(junit))
        clean = counts and counts["failures"] == counts["errors"] == counts["skipped"] == 0
        unchanged = source_digest(root) == run["source_sha256"] and tests_digest(root) == run["tests_sha256"]
        if exit_code == 0 and clean and unchanged:
            record["status"] = "passed"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
