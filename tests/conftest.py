"""Report which aophoton implementation the test session imported.

``scripts/run_tests.py`` sets ``AOPHOTON_IMPORT_RECORD`` and binds its evidence
to the reported module, so a passing record cannot certify the local source
while the tests actually exercised another installation.
"""

import os
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    target = os.environ.get("AOPHOTON_IMPORT_RECORD")
    if target:
        import aophoton

        Path(target).write_text(str(Path(aophoton.__file__).resolve()))
