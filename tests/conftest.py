"""Report what the test session imported, selected and concluded.

``scripts/run_tests.py`` sets ``AOPHOTON_SESSION_RECORD`` and binds its evidence
to this report: a passing record cannot certify the local source while the tests
exercised another installation, nor certify the full suite when only a selection
of it ran (``-k``, ``--deselect`` or ``--lf`` via ``PYTEST_ADDOPTS``, say).
"""

import json
import os
from pathlib import Path

_SESSION = {"selected": [], "deselected": 0, "outcomes": {}}


def pytest_collection_finish(session):
    _SESSION["selected"] = [item.nodeid for item in session.items]


def pytest_deselected(items):
    _SESSION["deselected"] += len(items)


def pytest_runtest_logreport(report):
    # A test passes only if its setup, call and teardown all pass; an expected
    # failure or unexpected pass is not a pass.
    outcomes = _SESSION["outcomes"]
    if report.failed:
        outcomes[report.nodeid] = "failed" if report.when == "call" else "error"
    elif report.skipped:
        outcomes.setdefault(report.nodeid, "xfailed" if hasattr(report, "wasxfail") else "skipped")
    elif report.when == "call":
        outcomes.setdefault(report.nodeid, "xpassed" if hasattr(report, "wasxfail") else "passed")


def pytest_sessionfinish(session, exitstatus):
    target = os.environ.get("AOPHOTON_SESSION_RECORD")
    if target:
        import aophoton

        Path(target).write_text(json.dumps(dict(_SESSION, imported_from=str(Path(aophoton.__file__).resolve()))))
