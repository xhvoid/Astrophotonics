"""Validation evidence cannot outlive a failed or stale rerun, or vouch for
inputs, products or code other than those that actually produced it."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_products(directory, skip=(), tag="old"):
    evidence = _load("evidence")
    for relative in evidence.REQUIRED_PRODUCTS:
        if relative in skip:
            continue
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{tag} {relative}\n")


@pytest.fixture
def project(tmp_path):
    """A minimal project tree holding a cleanly executed notebook, its inputs,
    its products, and old evidence."""
    (tmp_path / "src/aophoton").mkdir(parents=True)
    (tmp_path / "src/aophoton/__init__.py").write_text("")
    (tmp_path / "src/aophoton/core.py").write_text("VALUE = 1\n")
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/baseline.yaml").write_text("ao: {residual_wfe_nm: 150.0}\n")
    _write_products(tmp_path)
    (tmp_path / "results/showcase_metrics.json").write_text('{"baseline": {"mean_smf_coupling": 0.4711}}\n')
    (tmp_path / "notebooks").mkdir()
    cell = nbformat.v4.new_code_cell("1 + 1", execution_count=1)
    cell.outputs = [nbformat.v4.new_output("execute_result", {"text/plain": "2"}, execution_count=1)]
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), tmp_path / "notebooks/eso_astrophotonics_showcase.ipynb")
    (tmp_path / "results/notebook_execution.json").write_text(json.dumps(
        dict(status="passed", code_cells=12, error_outputs=0, png_figures=9, pdf_figures=9, wall_time_s=1.0)))
    return tmp_path


def _passing_notebook_record(evidence, project):
    return dict(status="passed", error_outputs=0,
                notebook_sha256=evidence.file_digest(project / evidence.NOTEBOOK),
                source_sha256=evidence.source_digest(project), inputs_sha256=evidence.inputs_digest(project),
                imported_from=evidence.IMPORT_TARGET, products_sha256=evidence.product_digests(project))


def test_failed_rerun_replaces_passing_record_and_blocks_doc_update(project, monkeypatch):
    execute = _load("execute_showcase")

    class FailingClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self, **kwargs):
            self.notebook.cells[0].outputs = [nbformat.v4.new_output(
                "error", ename="ValueError", evalue="boom", traceback=[])]
            raise RuntimeError("cell failed")

    class NoKernel:
        has_kernel = False

    monkeypatch.setattr(execute, "NotebookClient", FailingClient)
    monkeypatch.setattr(execute, "KernelManager", NoKernel)
    published = (project / "notebooks/eso_astrophotonics_showcase.ipynb").read_bytes()
    with pytest.raises(RuntimeError, match="cell failed"):
        execute.main(project)
    record = json.loads((project / "results/notebook_execution.json").read_text())
    assert record["status"] == "failed" and "cell failed" in record["error"]
    # The partial notebook is kept for diagnosis under the run's ID; the
    # published notebook still belongs to the published products.
    assert record["diagnostic_notebook"] == f"{execute.FAILED_EXECUTIONS}/{record['run_id']}.ipynb"
    partial = nbformat.read(project / record["diagnostic_notebook"], as_version=4)
    assert [(o.ename, o.evalue) for o in partial.cells[0].outputs] == [("ValueError", "boom")]
    assert (project / "notebooks/eso_astrophotonics_showcase.ipynb").read_bytes() == published

    update = _load("update_result_sections")
    with pytest.raises(RuntimeError, match="did not pass"):
        update.main(project)


def test_passing_record_is_bound_to_notebook_and_source(project):
    evidence = _load("evidence")
    evidence.write_record(project, _passing_notebook_record(evidence, project))
    assert evidence.verified_record(project)["status"] == "passed"

    # The pre-hash record format (as in an older passing run) is not trusted.
    record = json.loads((project / evidence.RECORD).read_text())
    evidence.write_record(project, {k: v for k, v in record.items() if k != "notebook_sha256"})
    with pytest.raises(RuntimeError, match="Notebook changed"):
        evidence.verified_record(project)
    evidence.write_record(project, record)

    (project / "src/aophoton/core.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="source changed"):
        evidence.verified_record(project)


@pytest.mark.parametrize("change", ["config", "metrics", "figure", "removed_product"])
def test_notebook_record_is_bound_to_inputs_and_products(project, change):
    # A passing record must not vouch for a different baseline configuration or
    # for result products edited (or removed) after the run.
    evidence = _load("evidence")
    evidence.write_record(project, _passing_notebook_record(evidence, project))
    if change == "config":
        (project / "configs/baseline.yaml").write_text("ao: {residual_wfe_nm: 500.0}\n")
        match = "Configuration inputs changed"
    elif change == "metrics":
        (project / "results/showcase_metrics.json").write_text('{"baseline": {"mean_smf_coupling": 0.9999}}\n')
        match = "Result products"
    elif change == "figure":
        (project / "figures/core_design_map.png").write_bytes(b"other")
        match = "Result products"
    else:
        (project / "figures/core_design_map.pdf").unlink()
        match = "Result products"
    with pytest.raises(RuntimeError, match=match):
        evidence.verified_record(project)


def test_unowned_files_neither_invalidate_nor_join_the_record(project):
    # A user's own file next to the products is not something the run wrote: it
    # must not block the documentation, and the record must not vouch for it.
    evidence = _load("evidence")
    evidence.write_record(project, _passing_notebook_record(evidence, project))
    (project / "figures/alignment_scan.png").write_bytes(b"independently acquired")
    record = evidence.verified_record(project)
    assert "figures/alignment_scan.png" not in record["products_sha256"]
    assert evidence.unowned_files(project, record) == ["figures/alignment_scan.png"]


def test_record_from_another_installation_is_rejected(project):
    evidence = _load("evidence")
    record = dict(_passing_notebook_record(evidence, project), imported_from="/elsewhere/aophoton/__init__.py")
    evidence.write_record(project, record)
    with pytest.raises(RuntimeError, match="imported aophoton from"):
        evidence.verified_record(project)


def test_pytest_evidence_is_bound_to_source_tests_and_junit(project):
    evidence = _load("evidence")
    (project / "tests").mkdir()
    (project / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    junit = project / evidence.JUNIT
    junit.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
    record = dict(status="passed", tests=1, intended_tests=1,
                  collection_sha256=evidence.collection_digest(["tests/test_x.py::test_x"]),
                  source_sha256=evidence.source_digest(project),
                  inputs_sha256=evidence.inputs_digest(project), imported_from=evidence.IMPORT_TARGET,
                  tests_sha256=evidence.tests_digest(project), junit_sha256=evidence.file_digest(junit),
                  pytest_config_sha256=evidence.file_digest(project / evidence.PYTEST_CONFIG))
    (project / evidence.TEST_RECORD).write_text(json.dumps(record))
    assert evidence.verified_test_record(project)["tests"] == 1

    # A record that does not bind its collection (the earlier format) is not trusted ...
    (project / evidence.TEST_RECORD).write_text(json.dumps({k: v for k, v in record.items()
                                                            if k != "collection_sha256"}))
    with pytest.raises(RuntimeError, match="intended test collection"):
        evidence.verified_test_record(project)
    (project / evidence.TEST_RECORD).write_text(json.dumps(record))
    # ... nor one whose pytest configuration has since changed.
    config = (project / evidence.PYTEST_CONFIG).read_text()
    (project / evidence.PYTEST_CONFIG).write_text(config + 'addopts = "-k nothing"\n')
    with pytest.raises(RuntimeError, match="pyproject.toml changed"):
        evidence.verified_test_record(project)
    (project / evidence.PYTEST_CONFIG).write_text(config)

    # A stale passing junit file cannot vouch for a changed test suite ...
    (project / "tests/test_x.py").write_text("def test_x():\n    assert False\n")
    with pytest.raises(RuntimeError, match="Tests or scripts changed"):
        evidence.verified_test_record(project)
    (project / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    # ... nor for changed configuration inputs ...
    (project / "configs/baseline.yaml").write_text("ao: {residual_wfe_nm: 500.0}\n")
    with pytest.raises(RuntimeError, match="Configuration inputs changed"):
        evidence.verified_test_record(project)
    (project / "configs/baseline.yaml").write_text("ao: {residual_wfe_nm: 150.0}\n")
    # ... nor can an edited pytest.xml or a failed run.
    junit.write_text('<testsuites><testsuite tests="9" failures="0" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(RuntimeError, match="pytest.xml does not match"):
        evidence.verified_test_record(project)
    (project / evidence.TEST_RECORD).write_text(json.dumps(dict(record, status="failed")))
    with pytest.raises(RuntimeError, match="did not pass"):
        evidence.verified_test_record(project)


def test_local_import_env_wins_over_a_shadowing_installation(tmp_path):
    # Another aophoton earlier on PYTHONPATH must not be what a run certifies.
    evidence = _load("evidence")
    shadow = tmp_path / "shadow/aophoton"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")
    probe = [sys.executable, "-c", "import aophoton; print(aophoton.__file__)"]
    shadowed = subprocess.run(probe, env={**evidence.local_import_env(ROOT), "PYTHONPATH": str(tmp_path / "shadow")},
                              capture_output=True, text=True, check=True, cwd=tmp_path)
    assert Path(shadowed.stdout.strip()).resolve() == (shadow / "__init__.py").resolve()
    local = subprocess.run(probe, env=evidence.local_import_env(ROOT, {"PYTHONPATH": str(tmp_path / "shadow")}),
                           capture_output=True, text=True, check=True, cwd=tmp_path)
    assert evidence.import_location(ROOT, local.stdout.strip()) == evidence.IMPORT_TARGET


@pytest.mark.parametrize("own_package", [True, False])
def test_run_tests_certifies_only_the_project_it_imported(project, own_package):
    # End to end on a throwaway project: the pytest session reports the module
    # it imported (tests/conftest.py) and the record passes only for its own src.
    (project / "tests").mkdir()
    shutil.copy(ROOT / "tests/conftest.py", project / "tests/conftest.py")
    (project / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    if not own_package:
        shutil.rmtree(project / "src/aophoton")       # falls through to the installed package
    run_tests = _load("run_tests")
    assert run_tests.main(project) == (0 if own_package else 1)
    record = json.loads((project / "results/pytest_record.json").read_text())
    evidence = _load("evidence")
    if own_package:
        assert record["status"] == "passed" and record["imported_from"] == evidence.IMPORT_TARGET
        assert evidence.verified_test_record(project)["tests"] == 1
    else:
        assert record["status"] == "failed" and record["imported_from"] != evidence.IMPORT_TARGET


def _throwaway_suite(project, names):
    (project / "tests").mkdir(exist_ok=True)
    shutil.copy(ROOT / "tests/conftest.py", project / "tests/conftest.py")
    (project / "tests/test_x.py").write_text("".join(f"def {name}():\n    pass\n\n" for name in names))


@pytest.mark.parametrize("options,status", [("-k test_a", "partial"),
                                            ("--deselect tests/test_x.py::test_b", "partial"),
                                            ("-o addopts=-ktest_a", "partial"),
                                            ("--collect-only", "failed")])
def test_selected_subset_cannot_certify_the_full_suite(project, monkeypatch, options, status):
    # A caller's test selection may pass, but it is only "partial" (and a run
    # that executes nothing fails): the record must not certify the suite, and
    # the documentation update must stay blocked.
    _throwaway_suite(project, ["test_a", "test_b", "test_c"])
    monkeypatch.setenv("PYTEST_ADDOPTS", options)
    run_tests = _load("run_tests")
    assert run_tests.main(project) == 1
    record = json.loads((project / "results/pytest_record.json").read_text())
    assert record["intended_tests"] == 3 and record["status"] == status
    evidence = _load("evidence")
    evidence.write_record(project, _passing_notebook_record(evidence, project))
    with pytest.raises(RuntimeError, match="partial|did not pass"):
        _load("update_result_sections").main(project)


def test_full_run_certifies_a_suite_that_grows(project, monkeypatch):
    # The count is whatever the configuration collects: no fixed test number.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    run_tests = _load("run_tests")
    evidence = _load("evidence")
    for names in (["test_a", "test_b"], ["test_a", "test_b", "test_c", "test_d"]):
        _throwaway_suite(project, names)
        assert run_tests.main(project) == 0
        record = evidence.verified_test_record(project)
        assert record["tests"] == record["intended_tests"] == len(names)
        assert record["collection_sha256"] == evidence.collection_digest(f"tests/test_x.py::{n}" for n in names)


def test_skipped_test_is_not_a_pass(project, monkeypatch):
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    _throwaway_suite(project, ["test_a"])
    with open(project / "tests/test_x.py", "a") as handle:
        handle.write("import pytest\n\n@pytest.mark.skip\ndef test_skipped():\n    pass\n")
    assert _load("run_tests").main(project) == 1
    record = json.loads((project / "results/pytest_record.json").read_text())
    assert record["status"] == "failed" and record["not_passed"] == ["tests/test_x.py::test_skipped"]


@pytest.mark.parametrize("moment,outcome", [("before the run", "passed"), ("during the session", "passed"),
                                            ("during the session", "failed")])
def test_test_record_and_junit_replace_symlinks(project, monkeypatch, tmp_path_factory, moment, outcome):
    # Links from the pytest record and pytest.xml to outside files are replaced
    # by regular files, never written through.  Made before the run, they meet
    # the "running" write; made from inside the test session, as another
    # process could, they meet the JUnit publication and the final record write.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    evidence = _load("evidence")
    outside = tmp_path_factory.mktemp("outside")
    links = {project / name: outside / f"{Path(name).name}.user" for name in (evidence.TEST_RECORD, evidence.JUNIT)}
    for sentinel in links.values():
        sentinel.write_bytes(b"original user data\n")
    _throwaway_suite(project, ["test_a"])
    if moment == "before the run":
        for path, sentinel in links.items():
            path.unlink(missing_ok=True)
            path.symlink_to(sentinel)
    else:
        body = ["import os", "", "", "def test_a():"]
        for path, sentinel in links.items():
            body += [f"    if os.path.lexists({str(path)!r}):", f"        os.unlink({str(path)!r})",
                     f"    os.symlink({str(sentinel)!r}, {str(path)!r})"]
        if outcome == "failed":
            body.append("    assert False")
        (project / "tests/test_x.py").write_text("\n".join(body) + "\n")
    code = _load("run_tests").main(project)
    assert all(sentinel.read_bytes() == b"original user data\n" for sentinel in links.values())
    assert all(path.is_file() and not path.is_symlink() for path in links)
    record = json.loads((project / evidence.TEST_RECORD).read_text())
    assert record["status"] == outcome and code == (0 if outcome == "passed" else 1)
    assert record["junit_sha256"] == evidence.file_digest(project / evidence.JUNIT)
    if outcome == "passed":
        assert evidence.verified_test_record(project)["tests"] == 1
    else:
        assert record["failures"] == 1 and record["not_passed"] == ["tests/test_x.py::test_a"]


def test_refused_final_test_record_leaves_the_running_record(project, monkeypatch):
    # If the final record cannot be renamed into place, the error propagates,
    # the run's own "running" record (which certifies nothing) stays, and the
    # writer's temporary file is removed.  The pytest sessions run in their own
    # processes, so only this process's renames are refused.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    _throwaway_suite(project, ["test_a"])
    evidence = _load("evidence")
    record = project / evidence.TEST_RECORD
    real_replace = os.replace
    renames = []

    def refusing_the_final_record(source, target):
        if Path(target) == record:
            renames.append(json.loads(Path(source).read_text())["status"])
            if len(renames) == 2:
                raise OSError("rename refused")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", refusing_the_final_record)
    with pytest.raises(OSError, match="rename refused"):
        _load("run_tests").main(project)
    assert renames == ["running", "passed"]          # the fault was reached at the final write
    assert json.loads(record.read_text())["status"] == "running"
    assert not list(record.parent.glob("*.publish-tmp"))
    with pytest.raises(RuntimeError, match="did not pass"):
        evidence.verified_test_record(project)


# One real ``run_tests.main`` invocation in a process of its own.  It saves the
# exact JUnit bytes it publishes as ``<base>/<tag>.xml`` before they reach the
# shared path and, with "pause", stops right after that publication, before
# its final record, until ``<base>/resume`` exists.
_RUNNER = """\
import io, sys, time
from pathlib import Path

scripts, project, base, tag, pause, timeout = sys.argv[1:]
sys.path.insert(0, scripts)
import evidence, run_tests

project, base = Path(project), Path(base)
publish = run_tests.atomic_write


def observed(target, fill, *args, **kwargs):
    if Path(target) != project / evidence.JUNIT:
        return publish(target, fill, *args, **kwargs)
    captured = io.BytesIO()
    fill(captured)
    (base / f"{tag}.xml").write_bytes(captured.getvalue())
    publish(target, lambda stream: stream.write(captured.getvalue()), *args, **kwargs)
    if pause == "pause":
        (base / f"{tag}.paused").touch()
        deadline = time.monotonic() + float(timeout)
        while not (base / "resume").exists():
            if time.monotonic() > deadline:
                print(f"{tag} was never resumed", file=sys.stderr)
                sys.exit(3)
            time.sleep(0.02)


run_tests.atomic_write = observed
sys.exit(run_tests.main(project))
"""

# Each session records which runner it belongs to; AUDIT_FAIL names a runner
# whose test_a fails.
_AUDITED_SUITE = """\
import os


def test_a(record_property):
    record_property("audit_origin", os.environ["AUDIT_ORIGIN"])
    assert os.environ.get("AUDIT_FAIL") != os.environ["AUDIT_ORIGIN"]


def test_b():
    pass
"""
_WAITING = "Waiting for another test run"


def _audited_project(project):
    _throwaway_suite(project, [])
    (project / "tests/test_x.py").write_text(_AUDITED_SUITE)


def _runner(base, project, tag, pause=False, **env):
    """Start a runner process on ``project`` whose sessions record ``tag``."""
    driver = base / "runner.py"
    if not driver.exists():
        driver.write_text(_RUNNER)
    inherited = {key: value for key, value in os.environ.items() if key not in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")}
    with open(base / f"{tag}.out", "w") as out, open(base / f"{tag}.err", "w") as err:
        return subprocess.Popen(
            [sys.executable, "-B", str(driver), str(ROOT / "scripts"), str(project), str(base), tag,
             "pause" if pause else "run", str(TIMEOUT_S)],
            stdout=out, stderr=err, env={**inherited, "AUDIT_ORIGIN": tag, **env})


def _await(base, condition, what, *processes):
    """Wait, at most ``TIMEOUT_S``, for ``condition``; fail at once if a runner exits first."""
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        exited = [process.args[6] for process in processes if process.poll() is not None]
        if exited or time.monotonic() > deadline:
            logs = "\n".join(f"{path.name}: {path.read_text()[-1500:]}" for path in sorted(base.glob("*.err")))
            pytest.fail(f"{f'{exited} exited' if exited else 'timed out'} before {what}\n{logs}")
        time.sleep(0.02)


def _contended_pair(base, project, first, second):
    """Run ``first = (tag, env)`` held just after publishing its report, start
    ``second`` then, and release ``first`` once ``second`` reports waiting for
    the run lock.  Returns the shared record and report as they stood while
    ``second`` waited, and each runner's exit code."""
    evidence = _load("evidence")
    (held, held_env), (late, late_env) = first, second
    processes = [_runner(base, project, held, pause=True, **held_env)]
    try:
        _await(base, (base / f"{held}.paused").exists, f"{held} published its report", *processes)
        processes.append(_runner(base, project, late, **late_env))
        _await(base, lambda: _WAITING in (base / f"{late}.err").read_text(),
               f"{late} waited for {held}'s run lock", *processes)
        waiting = dict(record=json.loads((project / evidence.TEST_RECORD).read_text()),
                       report=(project / evidence.JUNIT).read_bytes())
        (base / "resume").touch()
        codes = {process.args[6]: process.wait(TIMEOUT_S) for process in processes}
    finally:
        (base / "resume").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    return waiting, codes


def _printed_record(base, tag):
    """The final record a runner printed: its own, whatever the shared file now holds."""
    out = (base / f"{tag}.out").read_text()
    return json.loads(out[out.rindex('{\n  "status"'):])


def _origin(report):
    return ET.fromstring(report).find(".//property[@name='audit_origin']").get("value")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def test_concurrent_runs_each_certify_their_own_report(project, tmp_path_factory):
    # Two runner processes run the same tests (same node IDs and counts) in one
    # project, and each session records its runner in the JUnit report.  A is
    # held just after publishing pytest.xml: the window in which B used to
    # publish its own report, leaving A's record certifying B's XML.  B must
    # wait for A's run lock, and every record must describe its own report.
    evidence = _load("evidence")
    base = tmp_path_factory.mktemp("runs")
    _audited_project(project)
    waiting, codes = _contended_pair(base, project, ("A", {}), ("B", {}))
    assert codes == {"A": 0, "B": 0}
    printed = {tag: _printed_record(base, tag) for tag in "AB"}
    reports = {tag: (base / f"{tag}.xml").read_bytes() for tag in "AB"}
    # While B waited it had written nothing: the shared files were still A's.
    assert waiting["record"]["status"] == "running" and waiting["record"]["run_id"] == printed["A"]["run_id"]
    assert waiting["report"] == reports["A"]
    assert printed["A"]["run_id"] != printed["B"]["run_id"] and _sha256(reports["A"]) != _sha256(reports["B"])
    for tag in "AB":
        assert _origin(reports[tag]) == tag
        assert printed[tag]["status"] == "passed" and printed[tag]["tests"] == printed[tag]["intended_tests"] == 2
        assert printed[tag]["junit_sha256"] == _sha256(reports[tag])
    # B ran last: run ID, counts, digest and the published XML are all B's.
    assert evidence.verified_test_record(project) == printed["B"]
    published = (project / evidence.JUNIT).read_bytes()
    assert published == reports["B"] and _origin(published) == "B"


@pytest.mark.parametrize("non_passing_last", [False, True], ids=["passing-run-last", "non-passing-run-last"])
@pytest.mark.parametrize("outcome", ["failed", "partial"])
def test_overlapping_failed_or_partial_run_keeps_its_own_evidence(project, monkeypatch, tmp_path_factory, outcome,
                                                                  non_passing_last):
    # From a valid prior report, a passing run P and a run N that fails a test or
    # selects part of the suite overlap: whichever starts first is held after
    # publishing its report until the other waits for the lock.  N keeps its own
    # status, exit code, counts and report, and the shared pair is the last run's.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setenv("AUDIT_ORIGIN", "prior")
    evidence = _load("evidence")
    base = tmp_path_factory.mktemp("runs")
    _audited_project(project)
    assert _load("run_tests").main(project) == 0
    prior = evidence.verified_test_record(project)
    pair = [("P", {}), ("N", {"AUDIT_FAIL": "N"} if outcome == "failed" else {"PYTEST_ADDOPTS": "-k test_a"})]
    _, codes = _contended_pair(base, project, *(pair if non_passing_last else pair[::-1]))
    assert codes == {"P": 0, "N": 1}
    printed = {tag: _printed_record(base, tag) for tag in "PN"}
    reports = {tag: (base / f"{tag}.xml").read_bytes() for tag in "PN"}
    assert printed["P"]["status"] == "passed" and printed["N"]["status"] == outcome
    for tag in "PN":
        assert _origin(reports[tag]) == tag and printed[tag]["junit_sha256"] == _sha256(reports[tag])
        assert printed[tag]["run_id"] != prior["run_id"]
    counts = {key: printed["N"][key] for key in ("tests", "failures", "errors", "skipped", "selected_tests")}
    assert counts == (dict(tests=2, failures=1, errors=0, skipped=0, selected_tests=2) if outcome == "failed"
                      else dict(tests=1, failures=0, errors=0, skipped=0, selected_tests=1))
    last = "N" if non_passing_last else "P"
    assert json.loads((project / evidence.TEST_RECORD).read_text()) == printed[last]
    assert (project / evidence.JUNIT).read_bytes() == reports[last]
    if non_passing_last:
        with pytest.raises(RuntimeError, match="did not pass" if outcome == "failed" else "partial"):
            evidence.verified_test_record(project)
    else:
        assert evidence.verified_test_record(project) == printed["P"]


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_run_without_its_own_report_cannot_pass_on_another(project, monkeypatch, tmp_path_factory, damage):
    # A run whose pytest session passed but left no readable report of its own
    # fails, even with a valid report from an earlier passing run at the public
    # path: the session's last hook removes or truncates the private report and
    # puts that earlier report back at the public path, as another writer could.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.setenv("AUDIT_ORIGIN", "prior")
    evidence = _load("evidence")
    run_tests = _load("run_tests")
    _audited_project(project)
    assert run_tests.main(project) == 0
    prior = tmp_path_factory.mktemp("prior") / "pytest.xml"
    prior.write_bytes((project / evidence.JUNIT).read_bytes())
    damaging = "private.unlink()" if damage == "missing" else "private.write_bytes(private.read_bytes()[:60])"
    with open(project / "tests/conftest.py", "a") as conftest:
        conftest.write("\n\ndef pytest_unconfigure(config):\n"
                       "    import shutil\n    from pathlib import Path\n"
                       "    if config.option.xmlpath:\n"
                       "        private = Path(config.option.xmlpath)\n"
                       f"        {damaging}\n"
                       f"        shutil.copyfile({str(prior)!r}, {str(project / evidence.JUNIT)!r})\n")
    assert run_tests.main(project) == 1
    record = json.loads((project / evidence.TEST_RECORD).read_text())
    assert record["status"] == "failed" and record["exit_code"] == 0 and record["not_passed"] == []
    assert "tests" not in record
    published = (project / evidence.JUNIT).read_bytes()
    if damage == "missing":
        assert "junit_sha256" not in record and published == prior.read_bytes()      # never read back
    else:
        assert record["junit_sha256"] == _sha256(published) and published != prior.read_bytes()
    with pytest.raises(RuntimeError, match="did not pass"):
        evidence.verified_test_record(project)


def test_runs_of_different_projects_do_not_wait_for_each_other(project, tmp_path_factory):
    # The run lock is per project: while one checkout's run is held inside its
    # lock, a run of another checkout completes without waiting.
    evidence = _load("evidence")
    base = tmp_path_factory.mktemp("runs")
    _audited_project(project)
    other = tmp_path_factory.mktemp("unrelated") / "project"
    shutil.copytree(project, other)
    processes = [_runner(base, project, "A", pause=True)]
    try:
        _await(base, (base / "A.paused").exists, "A published its report", *processes)
        processes.append(_runner(base, other, "C"))
        assert processes[1].wait(TIMEOUT_S) == 0
        (base / "resume").touch()
        assert processes[0].wait(TIMEOUT_S) == 0
    finally:
        (base / "resume").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert _WAITING not in (base / "C.err").read_text()
    assert evidence.verified_test_record(other)["run_id"] == _printed_record(base, "C")["run_id"]
    assert evidence.verified_test_record(project)["run_id"] == _printed_record(base, "A")["run_id"]


class _NoKernel:
    has_kernel = False


def _staging_client(write):
    """A stand-in NotebookClient that 'executes' by calling ``write(stage)``,
    where ``stage`` is the kernel working directory the runner chose."""
    class StagingClient:
        def __init__(self, notebook, resources=None, **kwargs):
            self.stage = Path(resources["metadata"]["path"])

        def execute(self, **kwargs):
            write(self.stage)

    return StagingClient


def test_run_omitting_an_export_fails_and_keeps_published_products(project, monkeypatch):
    # The notebook computes everything but no longer writes showcase_metrics.json:
    # the older published copy must not be certified as this run's product.
    execute = _load("execute_showcase")
    before = (project / "results/showcase_metrics.json").read_text()
    monkeypatch.setattr(execute, "NotebookClient", _staging_client(
        lambda stage: _write_products(stage, skip={"results/showcase_metrics.json"}, tag="new")))
    monkeypatch.setattr(execute, "KernelManager", _NoKernel)
    monkeypatch.setattr(execute, "_kernel_import_path", lambda kernel: str(project / "src/aophoton/__init__.py"))
    with pytest.raises(RuntimeError, match="did not produce required outputs.*showcase_metrics.json"):
        execute.main(project)
    record = json.loads((project / "results/notebook_execution.json").read_text())
    assert record["status"] == "failed"
    assert (project / "results/showcase_metrics.json").read_text() == before
    assert (project / "figures/core_design_map.png").read_text().startswith("old")


def _own(evidence, project, paths):
    """Write an ownership manifest saying an earlier run wrote ``paths`` as they are now."""
    (project / evidence.OWNERSHIP).write_text(json.dumps(dict(
        run_id="earlier", products_sha256={p: evidence.file_digest(project / p) for p in paths})))


def _fake_notebook(execute, project, monkeypatch, write=lambda stage: _write_products(stage, tag="new")):
    monkeypatch.setattr(execute, "NotebookClient", _staging_client(write))
    monkeypatch.setattr(execute, "KernelManager", _NoKernel)
    monkeypatch.setattr(execute, "_kernel_import_path", lambda kernel: str(project / "src/aophoton/__init__.py"))


def test_complete_run_retires_its_own_products_and_keeps_user_files(project, monkeypatch):
    # All 25 products plus a retired figure an earlier run wrote, next to a
    # user's own alignment scan: the retired product goes, the scan survives
    # byte for byte, and the record certifies exactly this run's products.
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    (project / "figures/retired_plot.png").write_text("stale product of an earlier notebook\n")
    _own(evidence, project, [*evidence.REQUIRED_PRODUCTS, "figures/retired_plot.png"])
    scan = bytes(range(256)) * 3
    (project / "figures/alignment_scan.png").write_bytes(scan)
    (project / "results/showcase_baseline/notes.txt").write_text("bench log\n")
    _fake_notebook(execute, project, monkeypatch)
    execute.main(project)
    assert not (project / "figures/retired_plot.png").exists()
    assert (project / "figures/alignment_scan.png").read_bytes() == scan
    assert (project / "results/showcase_baseline/notes.txt").read_text() == "bench log\n"
    assert all((project / r).read_text().startswith("new") for r in evidence.REQUIRED_PRODUCTS)
    record = evidence.verified_record(project)
    assert record["status"] == "passed" and set(record["products_sha256"]) == set(evidence.REQUIRED_PRODUCTS)
    assert evidence.owned_products(project) == record["products_sha256"]
    assert not list(project.glob(".publish-backup-*"))


@pytest.mark.parametrize("ownership", ["legacy_record", "edited_since", "no_evidence", "corrupt_manifest",
                                       "outside_products"])
def test_retirement_needs_unambiguous_ownership(project, monkeypatch, ownership):
    # Without a manifest, a legacy passing record still identifies what its run
    # wrote; a file edited since, a record that did not pass, an unreadable
    # manifest or a path outside the product locations is kept.
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    stale = project / "figures/retired_plot.png"
    stale.write_text("stale product of an earlier notebook\n")
    outside = project / "src/aophoton/core.py"
    before = outside.read_text()
    legacy = dict(_passing_notebook_record(evidence, project), products_sha256=dict(
        evidence.product_digests(project), **{"src/aophoton/core.py": evidence.file_digest(outside)}))
    if ownership in ("legacy_record", "edited_since"):
        evidence.write_record(project, legacy)
        if ownership == "edited_since":
            stale.write_text("annotated by hand after that run\n")
    elif ownership == "corrupt_manifest":
        (project / evidence.OWNERSHIP).write_text("{not json")
    elif ownership == "outside_products":
        (project / evidence.OWNERSHIP).write_text(json.dumps(dict(products_sha256={
            "src/aophoton/core.py": evidence.file_digest(outside), "../escape.png": "0" * 64})))
    _fake_notebook(execute, project, monkeypatch)
    execute.main(project)
    assert stale.exists() == (ownership != "legacy_record")
    assert outside.read_text() == before
    assert evidence.verified_record(project)["status"] == "passed"


def test_audit_reproduction_keeps_an_unrelated_scan(tmp_path):
    # The review's reproduction: the full product set beside a user scan, with
    # the publisher called directly and no ownership evidence at all.
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    root, stage = tmp_path / "checkout", tmp_path / "stage"
    _write_products(root)
    _write_products(stage, tag="new")
    scan = root / "figures/alignment_scan.png"
    scan.write_bytes(b"independently acquired")
    execute._publish(stage, root, evidence.product_digests(stage))
    assert scan.read_bytes() == b"independently acquired"
    assert len(evidence.owned_products(root)) == len(evidence.REQUIRED_PRODUCTS)
    assert not list(root.glob(".publish-backup-*"))


def test_undeclared_product_cannot_overwrite_a_file_no_run_wrote(project, monkeypatch):
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    (project / "figures/extra.png").write_bytes(b"user figure")
    before = evidence.product_digests(project)

    def write(stage):
        _write_products(stage, tag="new")
        (stage / "figures/extra.png").write_bytes(b"generated")

    _fake_notebook(execute, project, monkeypatch, write)
    with pytest.raises(FileExistsError, match="nothing was changed"):
        execute.main(project)
    assert evidence.product_digests(project) == before


def test_staging_keeps_relative_templates_outside_data_working(tmp_path):
    # observation.template_path is relative to its YAML; a template outside data/
    # must still load when the notebook runs from the staging directory.
    import numpy as np

    from aophoton.config import load_config
    from aophoton.spectrum import load_spectrum, synthetic_spectrum

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "templates").mkdir()
    spectrum = synthetic_spectrum()
    np.savetxt(root / "templates/template.csv",
               np.column_stack((spectrum.wavelength_um, spectrum.relative_flux)), delimiter=",")
    (root / "configs/baseline.yaml").write_text("observation:\n  template_path: ../templates/template.csv\n")
    stage = tmp_path / "stage"
    stage.mkdir()
    execute._stage_inputs(root, stage)
    staged = load_config(stage / "configs/baseline.yaml")
    assert Path(staged.observation.template_path) == (root / "templates/template.csv").resolve()
    load_spectrum(staged.observation.template_path)
    assert not (stage / "results").exists() and not (stage / "figures").exists()   # outputs stay fresh
    # The referenced template is a run input: editing it invalidates the evidence.
    before = evidence.inputs_digest(root)
    (root / "templates/template.csv").write_text((root / "templates/template.csv").read_text() + "\n")
    assert evidence.inputs_digest(root) != before


def test_failed_publication_restores_the_previous_products_exactly(project, monkeypatch):
    # A write failure after several products were already replaced (and with a
    # stale extra pending removal) must leave the previous published set intact.
    import os

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    (project / "figures/retired_plot.png").write_text("stale product of an earlier notebook\n")
    _own(evidence, project, [*evidence.REQUIRED_PRODUCTS, "figures/retired_plot.png"])
    (project / "figures/alignment_scan.png").write_bytes(b"independently acquired")
    before = evidence.product_digests(project)
    manifest = (project / evidence.OWNERSHIP).read_text()
    calls, real_replace = [], os.replace

    def flaky_replace(source, target):
        calls.append(target)
        if len(calls) == 6:
            raise OSError("simulated write failure")
        return real_replace(source, target)

    _fake_notebook(execute, project, monkeypatch)
    monkeypatch.setattr(execute.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated write failure"):
        execute.main(project)
    assert len(calls) > 6                                    # some files were replaced, then restored
    assert evidence.product_digests(project) == before
    assert (project / evidence.OWNERSHIP).read_text() == manifest
    assert json.loads((project / "results/notebook_execution.json").read_text())["status"] == "failed"
    assert not list(project.glob(".publish-backup-*")) and not list(project.rglob("*.publish-tmp"))


def test_failure_after_retirement_restores_the_retired_product(project, monkeypatch):
    # The manifest is written after the retired product is removed: failing
    # there must bring the retired product back and leave no manifest behind.
    import os

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    (project / "figures/retired_plot.png").write_text("stale product of an earlier notebook\n")
    evidence.write_record(project, _passing_notebook_record(evidence, project))      # legacy ownership
    before = evidence.product_digests(project)
    real_replace = os.replace

    def failing_manifest(source, target):
        if Path(target).name == Path(evidence.OWNERSHIP).name:
            raise OSError("manifest write failure")
        return real_replace(source, target)

    _fake_notebook(execute, project, monkeypatch)
    monkeypatch.setattr(execute.os, "replace", failing_manifest)
    with pytest.raises(OSError, match="manifest write failure"):
        execute.main(project)
    assert evidence.product_digests(project) == before
    assert not (project / evidence.OWNERSHIP).exists()


def test_read_only_destination_is_refused_before_anything_changes(project, monkeypatch):
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    before = evidence.product_digests(project)
    locked = project / "results/showcase_metrics.json"
    locked.chmod(0o444)
    try:
        monkeypatch.setattr(execute, "NotebookClient", _staging_client(lambda stage: _write_products(stage, tag="new")))
        monkeypatch.setattr(execute, "KernelManager", _NoKernel)
        monkeypatch.setattr(execute, "_kernel_import_path", lambda kernel: str(project / "src/aophoton/__init__.py"))
        with pytest.raises(PermissionError, match="nothing was changed"):
            execute.main(project)
        assert evidence.product_digests(project) == before
    finally:
        locked.chmod(0o644)


def test_failed_rollback_names_the_retained_backup(project, monkeypatch):
    # Product storage that fails during publication AND restoration: the
    # complete previous set must survive in a backup that the error (and record)
    # names.  The record's own location keeps working, so the failure can be
    # recorded (a record that cannot be written is covered separately).
    import os

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    before = evidence.product_digests(project)
    calls, real_replace = [], os.replace

    def failing_storage(source, target):
        if Path(target).name != Path(evidence.RECORD).name:
            calls.append(target)
            if len(calls) > 2:
                raise OSError("persistent destination storage failure")
        return real_replace(source, target)

    monkeypatch.setattr(execute, "NotebookClient", _staging_client(lambda stage: _write_products(stage, tag="new")))
    monkeypatch.setattr(execute, "KernelManager", _NoKernel)
    monkeypatch.setattr(execute, "_kernel_import_path", lambda kernel: str(project / "src/aophoton/__init__.py"))
    monkeypatch.setattr(execute.os, "replace", failing_storage)
    with pytest.raises(RuntimeError, match="previous product set is preserved in") as raised:
        execute.main(project)
    [backup] = project.glob(".publish-backup-*")
    assert str(backup) in str(raised.value)
    assert evidence.product_digests(backup) == before                  # complete, unmodified previous set
    record = json.loads((project / "results/notebook_execution.json").read_text())
    assert record["status"] == "failed" and str(backup) in record["error"]


def test_moved_project_with_internal_and_external_templates_keeps_its_evidence(tmp_path):
    # Unchanged inputs must hash identically after the project moves, even when
    # one template is project-internal and another is an absolute external file.
    evidence = _load("evidence")
    shared = tmp_path / "m-shared"
    shared.mkdir()
    (shared / "external.csv").write_text("1.54,1.0\n1.57,1.0\n")
    source = tmp_path / "a-source"
    (source / "configs").mkdir(parents=True)
    (source / "templates").mkdir()
    (source / "templates/star.csv").write_text("1.54,0.9\n1.57,0.9\n")
    (source / "configs/baseline.yaml").write_text("observation:\n  template_path: ../templates/star.csv\n")
    (source / "configs/shared.yaml").write_text(f"observation:\n  template_path: {shared / 'external.csv'}\n")
    before = evidence.inputs_digest(source)
    moved = tmp_path / "z-moved"
    source.rename(moved)
    assert evidence.inputs_digest(moved) == before
    (shared / "external.csv").write_text("1.54,0.5\n1.57,0.5\n")      # a real change is still detected
    assert evidence.inputs_digest(moved) != before


def test_generated_labels_follow_the_run_not_the_script(project, monkeypatch):
    # A run with 3 modes, 3 lantern ports and 80% transmission on an 8 m pupil:
    # every label in the refreshed documents must come from its own metadata.
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    run_configuration = dict(
        telescope=dict(diameter_m=8.0), ao=dict(wavelength_um=1.60),
        fibre=dict(few_mode_count=3, lantern_ports=3, lantern_transmission=0.8, feed="lantern"),
        spectrograph=dict(wavelength_min_um=1.58, wavelength_max_um=1.62, resolving_power=80_000.0),
        detector=dict(extraction_pixels=4))
    baseline = dict(residual_wfe_nm=150.0, measured_strehl=0.5, marechal_without_jitter=0.6, mean_smf_coupling=0.4,
                    mean_few_mode_coupling=0.5, mean_lantern_coupling=0.45, median_snr_per_spectral_sample=120.0,
                    median_snr_per_pixel=120.0, detector_pixels_per_spectral_sample=12, noise_mode="noisy",
                    injected_rv_m_s=2300.0, recovered_rv_m_s=2301.0, recovery_bias_m_s=1.0,
                    estimated_rv_uncertainty_m_s=2.0, rv_reduced_chi2=1.01)
    metrics = dict(baseline=baseline, run_configuration=run_configuration, conclusions=["Derived."],
                   strehl_cases=[[0, 1.0, 1.0], [150, 0.69, 0.69], [400, 0.21, 0.07]],
                   sweep_frames=12, maximum_coupling_sem=0.01)
    trade = [dict(brightness_scale=b, wfe_nm=w, ports=n, rv_sigma_m_s=s)
             for b, w, n, s in [(1.0, 75.0, 1, 2.0), (1.0, 75.0, 3, 1.5), (1.0, 300.0, 1, 3.0), (1.0, 300.0, 3, 2.94),
                                (0.01, 300.0, 1, 30.0), (0.01, 300.0, 3, 29.99)]]
    (project / "results/showcase_metrics.json").write_text(json.dumps(metrics))
    (project / "results/port_trade.json").write_text(json.dumps(trade))
    for name in ("README.md", "docs/technical_note.md"):
        (project / name).parent.mkdir(exist_ok=True)
        (project / name).write_text("Intro\n<!-- BEGIN GENERATED RESULTS -->\nold\n<!-- END GENERATED RESULTS -->\nEnd\n")
    _throwaway_suite(project, ["test_a"])
    assert _load("run_tests").main(project) == 0
    evidence = _load("evidence")
    evidence.write_record(project, dict(_passing_notebook_record(evidence, project), code_cells=12, png_figures=9,
                                        pdf_figures=9, wall_time_s=1.0, run_id="0" * 32,
                                        python="3.13.1 (main, Jan 1 2026) [Clang]"))
    scan = project / "figures/alignment_scan.png"
    scan.write_bytes(b"bench data")
    update = _load("update_result_sections")
    update.main(project)
    readme = (project / "README.md").read_text()
    for label in ("Mean ideal 3-mode acceptance", "Mean 3-port lantern coupling, after 80% transmission",
                  "Median SNR per extracted spectral sample (12 detector pixels) | 120.00",
                  "8 m pupil, 1.580–1.620 µm band (AO reference 1.600 µm), R = 80,000, lantern feed, noisy",
                  "(Python 3.13.1; CPU only)"):
        assert label in readme
    for stale in ("six", "90%", "per detector pixel", "3.14.3", "3.6 m"):
        assert stale not in readme
    note = (project / "docs/technical_note.md").read_text()
    assert "SMF and 3-port forecasts are closest at 300 nm WFE (3.00 versus 2.94 m/s, 2.0% apart)" in note
    assert "150 nm bright case" not in note
    # The unrelated scan is reported, untouched and not certified.
    assert scan.read_bytes() == b"bench data"
    assert "figures/alignment_scan.png" not in evidence.verified_record(project)["products_sha256"]


# Content reads recorded by an audit hook (audit hooks cannot be removed, so one
# hook is installed for the session and switched on only while a test listens).
_OPENED: list | None = None


def _record_open(event, args):
    if _OPENED is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.path.realpath(os.fsdecode(args[0])))


sys.addaudithook(_record_open)


def test_publication_reads_no_unrelated_file(project, monkeypatch):
    # An unreadable personal plot and a bench log in a product location: a
    # complete run publishes, verifies and lists them without opening either,
    # which the audit hook checks even where permissions do not apply (root).
    global _OPENED
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    _own(evidence, project, evidence.REQUIRED_PRODUCTS)
    personal = project / "figures/personal_plot.png"
    personal.write_bytes(b"private plot")
    notes = project / "results/showcase_baseline/notes.txt"
    notes.write_bytes(b"bench log")
    unrelated = {os.path.realpath(path) for path in (personal, notes)}
    _fake_notebook(execute, project, monkeypatch)
    personal.chmod(0)
    _OPENED = []
    try:
        execute.main(project)
        record = evidence.verified_record(project)
        listed = evidence.unowned_files(project, record)
        opened = set(_OPENED)
    finally:
        _OPENED = None
        personal.chmod(0o600)
    assert opened, "the audit hook saw no reads at all"
    assert str(project / "results/showcase_metrics.json") in {p.replace(os.path.realpath(project), str(project))
                                                              for p in opened}
    assert not opened & unrelated
    assert listed == ["figures/personal_plot.png", "results/showcase_baseline/notes.txt"]
    assert personal.read_bytes() == b"private plot" and notes.read_bytes() == b"bench log"
    assert all((project / r).read_text().startswith("new") for r in evidence.REQUIRED_PRODUCTS)


@pytest.mark.parametrize("collision", ["file", "symlink"])
@pytest.mark.parametrize("failure", [None, "manifest"])
def test_temporary_files_never_reuse_an_existing_name(tmp_path, monkeypatch, collision, failure):
    # A regular file or a link to an outside sentinel at each temporary name the
    # publisher used to reuse: both survive, the sentinel is untouched, and the
    # published product and manifest are regular files with the staged content.
    # After a failed manifest write, only that publisher's own temporaries go.
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    root, stage = tmp_path / "checkout", tmp_path / "stage"
    _write_products(root)
    _write_products(stage, tag="new")
    (root / evidence.OWNERSHIP).write_text('{"run_id": "earlier", "products_sha256": {}}\n')
    sentinel = tmp_path / "outside/sentinel.bin"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"private original data")
    former = [root / "figures/.core_design_map.png.publish-tmp", root / "results/.notebook_products.json.publish-tmp"]
    for path in former:
        if collision == "file":
            path.write_bytes(b"unrelated " + path.name.encode())
        else:
            path.symlink_to(sentinel)
    kept = {path: os.readlink(path) if collision == "symlink" else path.read_bytes() for path in former}
    before = {relative: (root / relative).read_bytes() for relative in [*evidence.REQUIRED_PRODUCTS, evidence.OWNERSHIP]}
    if failure:
        real_replace = os.replace

        def failing_manifest(source, target):
            if Path(target).name == Path(evidence.OWNERSHIP).name:
                raise OSError("manifest write failure")
            return real_replace(source, target)

        monkeypatch.setattr(execute.os, "replace", failing_manifest)
        with pytest.raises(OSError, match="manifest write failure"):
            execute._publish(stage, root, evidence.product_digests(stage), run_id="new")
        assert {relative: (root / relative).read_bytes() for relative in before} == before
    else:
        execute._publish(stage, root, evidence.product_digests(stage), run_id="new")
        for relative in ("figures/core_design_map.png", evidence.OWNERSHIP):
            assert (root / relative).is_file() and not (root / relative).is_symlink()
        assert (root / "figures/core_design_map.png").read_bytes() == (stage / "figures/core_design_map.png").read_bytes()
        assert json.loads((root / evidence.OWNERSHIP).read_text())["run_id"] == "new"
    assert sentinel.read_bytes() == b"private original data"
    for path, original in kept.items():
        assert path.is_symlink() == (collision == "symlink")
        assert (os.readlink(path) if collision == "symlink" else path.read_bytes()) == original
    assert sorted(root.rglob("*.publish-tmp")) == sorted(former)
    assert not list(root.glob(".publish-backup-*"))


def test_failed_copy_removes_only_its_own_temporary(tmp_path, monkeypatch):
    execute = _load("execute_showcase")
    target = tmp_path / "figures/core_design_map.png"
    target.parent.mkdir()
    target.write_bytes(b"published")
    former = target.with_name(".core_design_map.png.publish-tmp")
    former.write_bytes(b"unrelated")
    source = tmp_path / "staged.png"
    source.write_bytes(b"x" * 100_000)

    def interrupted(original, stream, *args, **kwargs):
        stream.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(execute.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="disk full"):
        execute._replace_file(source, target)
    assert target.read_bytes() == b"published" and former.read_bytes() == b"unrelated"
    assert sorted(path.name for path in target.parent.iterdir()) == sorted([target.name, former.name])


def _generations(execute, project, monkeypatch, behaviour):
    """Fake notebook runs, one per thread: thread ``name`` writes products tagged
    ``name`` (plus ``figures/only_<name>.png``), marks the executed notebook
    with its generation, then calls ``behaviour[name]()``.  If that raises, the
    cell keeps the error as its output before the error propagates, as
    nbclient does."""
    def write(stage):
        import threading
        name = threading.current_thread().name
        _write_products(stage, tag=name)
        (stage / f"figures/only_{name}.png").write_text(f"{name} only\n")
        behaviour[name]()

    class GenerationClient:
        def __init__(self, notebook, resources=None, **kwargs):
            self.notebook, self.stage = notebook, Path(resources["metadata"]["path"])

        def execute(self, **kwargs):
            import threading
            self.notebook.metadata["generation"] = threading.current_thread().name
            try:
                write(self.stage)
            except Exception as error:
                self.notebook.cells[0].outputs = [nbformat.v4.new_output(
                    "error", ename=type(error).__name__, evalue=str(error), traceback=[])]
                raise

    monkeypatch.setattr(execute, "NotebookClient", GenerationClient)
    monkeypatch.setattr(execute, "KernelManager", _NoKernel)
    monkeypatch.setattr(execute, "_kernel_import_path", lambda kernel: str(project / "src/aophoton/__init__.py"))


def _start(target, name, errors):
    import threading

    def run():
        try:
            target()
        except BaseException as error:          # reported by the test thread
            errors[name] = error

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread


def _published_generation(evidence, project, name):
    """Assert the complete published state belongs to generation ``name``."""
    produced = {relative: evidence.file_digest(project / relative)
                for relative in [*evidence.REQUIRED_PRODUCTS, f"figures/only_{name}.png"]}
    assert all((project / r).read_text().startswith(name) for r in evidence.REQUIRED_PRODUCTS)
    assert evidence.owned_products(project) == produced
    assert nbformat.read(project / evidence.NOTEBOOK, as_version=4).metadata["generation"] == name
    stray = {r for r in evidence.product_paths(project) if r.startswith("figures/only_")} - set(produced)
    assert not stray, f"another generation's products remain: {stray}"
    return json.loads((project / evidence.OWNERSHIP).read_text())["run_id"]


TIMEOUT_S = 30


def test_overlapping_publishers_commit_whole_generations(project, monkeypatch):
    # A holds the publication lock in mid-commit while B, which started from the
    # same old generation and has finished executing, waits for it.  A commits,
    # then B commits; B's publication retires A's extra product, which A's
    # manifest (not B's stale start-up snapshot) says A owns.
    import threading

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    _own(evidence, project, evidence.REQUIRED_PRODUCTS)
    b_executed, a_committing, b_waiting, release_a = (threading.Event() for _ in range(4))
    _generations(execute, project, monkeypatch, {
        "A": lambda: b_executed.wait(TIMEOUT_S) or pytest.fail("B never executed"),
        "B": lambda: (b_executed.set(), a_committing.wait(TIMEOUT_S) or pytest.fail("A never committed")),
    })
    real_manifest = execute._write_ownership

    def pausing_manifest(root, produced, run_id):
        if threading.current_thread().name == "A":
            a_committing.set()
            assert release_a.wait(TIMEOUT_S), "never released"
        real_manifest(root, produced, run_id)

    monkeypatch.setattr(execute, "_write_ownership", pausing_manifest)
    monkeypatch.setattr(execute, "_report_wait", lambda root: threading.current_thread().name == "B" and b_waiting.set())
    errors = {}
    threads = [_start(lambda: execute.main(project), name, errors) for name in ("A", "B")]
    assert b_waiting.wait(TIMEOUT_S), "B never waited for A's lock"
    assert not (project / "figures/only_B.png").exists()           # B cannot publish inside A's commit
    release_a.set()
    for thread in threads:
        thread.join(TIMEOUT_S)
        assert not thread.is_alive(), f"{thread.name} did not finish"
    assert not errors, errors
    run_id = _published_generation(evidence, project, "B")
    record = evidence.verified_record(project)
    assert record["status"] == "passed" and record["run_id"] == run_id
    assert record["products_sha256"] == evidence.owned_products(project)


def test_failed_publisher_keeps_the_generation_committed_before_it(project, monkeypatch):
    # A starts from the old generation and is still executing when B executes,
    # publishes and returns; A's own publication then fails.  It must restore B's
    # generation (products, manifest and notebook), never the one it started
    # from, and report its own failure in the execution record.
    import threading

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    _own(evidence, project, evidence.REQUIRED_PRODUCTS)
    b_committed = threading.Event()
    _generations(execute, project, monkeypatch, {
        "A": lambda: b_committed.wait(TIMEOUT_S) or pytest.fail("B never committed"),
        "B": lambda: None,
    })
    real_manifest = execute._write_ownership

    def failing_for_a(root, produced, run_id):
        if threading.current_thread().name == "A":
            raise OSError("A's manifest write failed")
        real_manifest(root, produced, run_id)

    monkeypatch.setattr(execute, "_write_ownership", failing_for_a)
    errors = {}
    a = _start(lambda: execute.main(project), "A", errors)
    b = _start(lambda: execute.main(project), "B", errors)
    b.join(TIMEOUT_S)
    assert not b.is_alive() and "B" not in errors, errors
    b_record = evidence.verified_record(project)
    b_committed.set()
    a.join(TIMEOUT_S)
    assert not a.is_alive(), "A did not finish"
    assert isinstance(errors.pop("A"), OSError) and not errors
    assert _published_generation(evidence, project, "B") == b_record["run_id"]
    record = json.loads((project / evidence.RECORD).read_text())
    assert record["status"] == "failed" and record["run_id"] != b_record["run_id"]
    assert "A's manifest write failed" in record["error"]
    assert {r: evidence.file_digest(project / r) for r in b_record["products_sha256"]} == b_record["products_sha256"]
    assert evidence.file_digest(project / evidence.NOTEBOOK) == b_record["notebook_sha256"]


def _committed_generation(evidence, project):
    """A verified passing generation as it is installed: record, notebook and
    manifest bytes, and the digest of every product the record lists."""
    record = evidence.verified_record(project)
    return dict(record=record, notebook=(project / evidence.NOTEBOOK).read_bytes(),
                manifest=(project / evidence.OWNERSHIP).read_bytes(),
                products={r: evidence.file_digest(project / r) for r in record["products_sha256"]})


def _still_installed(evidence, project, committed, failed):
    """Assert ``committed`` is installed byte for byte and run ``failed`` left no product."""
    assert (project / evidence.NOTEBOOK).read_bytes() == committed["notebook"]
    assert (project / evidence.OWNERSHIP).read_bytes() == committed["manifest"]
    assert len(committed["products"]) == len(evidence.REQUIRED_PRODUCTS) + 1        # and its own extra
    present = {r: evidence.file_digest(project / r) for r in committed["products"] if (project / r).is_file()}
    assert present == committed["products"] == committed["record"]["products_sha256"]
    assert not (project / f"figures/only_{failed}.png").exists()
    assert evidence.unowned_files(project, committed["record"]) == []
    assert not list(project.glob(".publish-backup-*")) and not list(project.rglob("*.publish-tmp"))


def test_failed_success_record_restores_the_generation_committed_before_it(project, monkeypatch):
    # A is paused in execution while B executes, publishes and is verified.  A
    # then publishes different files, and the rename that commits A's passing
    # record fails.  Everything A replaced must be rolled back to B's generation
    # (notebook, manifest and all 26 products), and the record must report A's
    # failure, which certifies nothing.
    import threading

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    _own(evidence, project, evidence.REQUIRED_PRODUCTS)
    a_executing, b_committed = threading.Event(), threading.Event()
    _generations(execute, project, monkeypatch, {
        "A": lambda: (a_executing.set(), b_committed.wait(TIMEOUT_S) or pytest.fail("B never committed")),
        "B": lambda: None,
    })
    record_path, real_replace, at_fault = project / evidence.RECORD, os.replace, {}

    def failing_commit(source, target):
        # The commit boundary itself: A's passing record renamed onto the record.
        if (threading.current_thread().name == "A" and Path(target) == record_path
                and json.loads(Path(source).read_text())["status"] == "passed"):
            at_fault.update(
                only_a=(project / "figures/only_A.png").exists(),
                manifest_run=json.loads((project / evidence.OWNERSHIP).read_text())["run_id"],
                notebook=nbformat.read(project / evidence.NOTEBOOK, as_version=4).metadata["generation"],
                record_run=json.loads(record_path.read_text())["run_id"])
            raise OSError("injected failure committing A's success record")
        return real_replace(source, target)

    monkeypatch.setattr(execute.os, "replace", failing_commit)
    errors = {}
    a = _start(lambda: execute.main(project), "A", errors)
    assert a_executing.wait(TIMEOUT_S), "A never started executing"
    b = _start(lambda: execute.main(project), "B", errors)
    b.join(TIMEOUT_S)
    assert not b.is_alive() and "B" not in errors, errors
    committed = _committed_generation(evidence, project)
    b_committed.set()
    a.join(TIMEOUT_S)
    assert not a.is_alive(), "A did not finish"
    assert at_fault, "A never reached the commit of its success record"
    error = errors.pop("A")
    assert isinstance(error, OSError) and "injected failure committing" in str(error) and not errors
    record = json.loads(record_path.read_text())
    # When the commit failed, A's products, manifest and notebook were in place
    # and B's record was still the published one ...
    assert at_fault == dict(only_a=True, manifest_run=record["run_id"], notebook="A",
                            record_run=committed["record"]["run_id"])
    # ... and now B's generation is back, byte for byte, with A reported failed.
    _still_installed(evidence, project, committed, "A")
    assert record["status"] == "failed" and record["run_id"] != committed["record"]["run_id"]
    assert "injected failure committing" in record["error"]
    with pytest.raises(RuntimeError, match="did not pass"):
        evidence.verified_record(project)
    with pytest.raises(RuntimeError, match="did not pass"):
        _load("update_result_sections").main(project)


def test_older_run_failing_later_keeps_the_newer_published_notebook(project, monkeypatch):
    # A starts first and is paused inside a cell; B runs to completion and is
    # verified.  A's cell then fails, before A publishes anything.  B's notebook,
    # manifest and products stay installed byte for byte, A's partial notebook
    # is kept under A's run ID with A's error in it, and A is reported failed.
    import threading

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    _own(evidence, project, evidence.REQUIRED_PRODUCTS)
    a_in_cell, b_committed = threading.Event(), threading.Event()

    def a_cell():
        a_in_cell.set()
        assert b_committed.wait(TIMEOUT_S), "B never committed"
        raise RuntimeError("A's cell failed after B committed")

    _generations(execute, project, monkeypatch, {"A": a_cell, "B": lambda: None})
    publishers, real_publish = [], execute._publish

    def recording_publish(*args, **kwargs):
        publishers.append(threading.current_thread().name)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(execute, "_publish", recording_publish)
    errors = {}
    a = _start(lambda: execute.main(project), "A", errors)
    assert a_in_cell.wait(TIMEOUT_S), "A never entered its cell"
    b = _start(lambda: execute.main(project), "B", errors)
    b.join(TIMEOUT_S)
    assert not b.is_alive() and "B" not in errors, errors
    committed = _committed_generation(evidence, project)
    b_committed.set()
    a.join(TIMEOUT_S)
    assert not a.is_alive(), "A did not finish"
    error = errors.pop("A")
    assert isinstance(error, RuntimeError) and "A's cell failed" in str(error) and not errors
    assert publishers == ["B"]                      # A failed in execution, never reached publication
    _still_installed(evidence, project, committed, "A")
    assert evidence.file_digest(project / evidence.NOTEBOOK) == committed["record"]["notebook_sha256"]
    record = json.loads((project / evidence.RECORD).read_text())
    assert record["status"] == "failed" and record["run_id"] != committed["record"]["run_id"]
    assert record["diagnostic_notebook"] == f"{execute.FAILED_EXECUTIONS}/{record['run_id']}.ipynb"
    partial = nbformat.read(project / record["diagnostic_notebook"], as_version=4)
    assert partial.metadata["generation"] == "A"
    assert [(o.ename, o.evalue) for o in partial.cells[0].outputs] == [
        ("RuntimeError", "A's cell failed after B committed")]


@pytest.mark.parametrize("status", ["running", "passed", "failed"])
def test_execution_record_replaces_a_symlink_at_every_status(project, monkeypatch, tmp_path_factory, status):
    # A link from the record to an outside file must be replaced by a regular
    # file, never written through, by each of the runner's record writes.  The
    # link is (re)created just before the write under test, so an earlier safe
    # replacement cannot hide an unsafe later writer.
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    _own(evidence, project, evidence.REQUIRED_PRODUCTS)
    sentinel = tmp_path_factory.mktemp("outside") / "user_data.json"
    sentinel.write_bytes(b'{"owner": "someone else", "bytes": "original"}\n')
    original = sentinel.read_bytes()
    record = project / evidence.RECORD

    def link():
        record.unlink()
        record.symlink_to(sentinel)

    running = {}

    def cells(stage):
        running.update(regular=record.is_file() and not record.is_symlink(),
                       sentinel=sentinel.read_bytes() == original, **json.loads(record.read_text()))
        if status != "running":
            link()                                  # before the terminal write under test
        _write_products(stage, tag="new")
        if status == "failed":
            raise RuntimeError("cell failed")

    if status == "running":
        link()
    _fake_notebook(execute, project, monkeypatch, write=cells)
    if status == "failed":
        with pytest.raises(RuntimeError, match="cell failed"):
            execute.main(project)
    else:
        execute.main(project)
    final = json.loads(record.read_text())
    assert sentinel.read_bytes() == original
    assert record.is_file() and not record.is_symlink()
    assert running["regular"] and running["sentinel"] and running["status"] == "running"
    assert final["status"] == ("passed" if status == "running" else status)
    assert final["run_id"] == running["run_id"]


@pytest.mark.parametrize("failure", ["write", "rename"])
def test_failed_record_write_keeps_the_previous_record(tmp_path, monkeypatch, failure):
    # An interrupted write or a refused rename leaves the previous record byte
    # for byte and removes the writer's own temporary file, and nothing else.
    evidence = _load("evidence")
    record = tmp_path / evidence.RECORD
    record.parent.mkdir()
    evidence.write_record(tmp_path, dict(status="passed", run_id="previous"))
    previous = record.read_bytes()
    unrelated = record.with_name(f".{record.name}.publish-tmp")            # the name a fixed scheme would use
    unrelated.write_bytes(b"unrelated")
    real_fdopen = os.fdopen

    class Interrupted:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:10])
            self.stream.flush()
            raise OSError("disk full")

    def refused(source, target):
        raise OSError("rename refused")

    with monkeypatch.context() as patch:
        if failure == "write":
            patch.setattr(evidence.os, "fdopen", lambda fd, *args, **kwargs: Interrupted(real_fdopen(fd, *args, **kwargs)))
        else:
            patch.setattr(evidence.os, "replace", refused)
        with pytest.raises(OSError, match="disk full" if failure == "write" else "rename refused"):
            evidence.write_record(tmp_path, dict(status="failed", run_id="new"))
    assert record.read_bytes() == previous
    assert sorted(path.name for path in record.parent.iterdir()) == sorted([record.name, unrelated.name])
    assert unrelated.read_bytes() == b"unrelated"


def test_unrecordable_failure_keeps_a_non_passing_record_and_the_original_error(project, monkeypatch, capsys):
    # If the failed record cannot be written, the run's own running record stays
    # (it certifies nothing), the reason is printed, and the cell error, not the
    # record error, propagates.
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    real_replace = os.replace

    def refusing_failed_record(source, target):
        if Path(target).name == Path(evidence.RECORD).name and json.loads(Path(source).read_text())["status"] == "failed":
            raise OSError("record storage failure")
        return real_replace(source, target)

    def failing_cells(stage):
        raise RuntimeError("cell failed")

    _fake_notebook(execute, project, monkeypatch, write=failing_cells)
    monkeypatch.setattr(execute.os, "replace", refusing_failed_record)
    with pytest.raises(RuntimeError, match="cell failed"):
        execute.main(project)
    assert json.loads((project / evidence.RECORD).read_text())["status"] == "running"
    assert "Could not record this run's failure" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="did not pass"):
        evidence.verified_record(project)


def test_publication_waits_for_a_lock_held_by_another_process(tmp_path, monkeypatch):
    import threading

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    root, stage = tmp_path / "checkout", tmp_path / "stage"
    _write_products(root)
    _write_products(stage, tag="new")
    lock = root / execute.PUBLICATION_LOCK
    holder = subprocess.Popen(
        [sys.executable, "-c", "import fcntl, os, sys\n"
         f"fd = os.open({str(lock)!r}, os.O_RDWR | os.O_CREAT)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\nprint('locked', flush=True)\nsys.stdin.readline()\n"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        waiting = threading.Event()
        monkeypatch.setattr(execute, "_report_wait", lambda root: waiting.set())
        before = evidence.product_digests(root)
        errors = {}
        publisher = _start(lambda: execute._publish(stage, root, evidence.product_digests(stage), run_id="new"),
                           "publisher", errors)
        assert waiting.wait(TIMEOUT_S), "the publisher did not wait for the other process"
        assert evidence.product_digests(root) == before
        holder.stdin.write("\n")
        holder.stdin.flush()
        publisher.join(TIMEOUT_S)
        assert not publisher.is_alive() and not errors, errors
    finally:
        holder.kill() if holder.poll() is None else None
        holder.wait(TIMEOUT_S)
    assert all((root / r).read_text().startswith("new") for r in evidence.REQUIRED_PRODUCTS)


def _notebook_sweeps(tmp_path, monkeypatch, outer_scale_m):
    """Run the delivered notebook's code cells, in order, up to its wavelength
    and seeing scans, on a 32-pixel copy of the baseline with ``outer_scale_m``."""
    # Earlier tests' execute_showcase.main() leave MPLCONFIGDIR naming a deleted
    # temporary project; matplotlib would rebuild its font cache there.
    monkeypatch.setenv("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib
    import yaml

    matplotlib.use("Agg")
    root = tmp_path / f"L0_{outer_scale_m:g}"
    (root / "configs").mkdir(parents=True)
    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
    config["telescope"]["n_pix"] = 32
    config["ao"]["outer_scale_m"] = outer_scale_m
    (root / "configs/baseline.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.chdir(root)
    context = {"__name__": "__main__"}
    cells = [cell for cell in nbformat.read(ROOT / "notebooks/eso_astrophotonics_showcase.ipynb", as_version=4).cells
             if cell.cell_type == "code"]
    for index, cell in enumerate(cells):
        exec(compile(cell.source, f"<cell {index}>", "exec"), context)
        if index == 0:                                    # figures are not under test
            context["export"] = lambda fig, name: context["plt"].close(fig)
        if "seeing_eta" in context:
            return context
    raise AssertionError("the notebook no longer computes its seeing scan")


def test_notebook_sweeps_use_the_configured_outer_scale(tmp_path, monkeypatch):
    # The design map, wavelength and seeing scans (which feed the port trade)
    # used the default 25 m outer scale whatever the configuration said: at
    # 0.5 m the 400 nm / 10 mas SMF cell read 0.050513 instead of 0.051988.
    import numpy as np

    from aophoton.atmosphere import make_ao_residual
    from aophoton.coupling import overlap_efficiency
    from aophoton.lantern import mode_basis
    from aophoton.propagation import propagate
    from aophoton.pupil import make_pupil

    runs = {L0: _notebook_sweeps(tmp_path, monkeypatch, L0) for L0 in (25.0, 0.5)}
    expected = {25.0: 0.050513, 0.5: 0.051988}
    for L0, context in runs.items():
        cfg = context["cfg"]
        assert cfg.ao.outer_scale_m == L0
        t = cfg.telescope
        pupil = make_pupil(t.n_pix, t.diameter_m, t.obscuration, t.spider_width_m)
        reference = propagate(pupil, np.zeros_like(pupil.amplitude), cfg.ao.wavelength_um, t.pad_factor)
        mode = mode_basis(reference, cfg.fibre.mode_radius_mas, len(context["basis"]))[0]
        frames = context["sweep_frames"]
        tilts = np.random.default_rng(810).normal(size=(frames, 2))
        iw = int(np.flatnonzero(context["wfe_grid"] == 400)[0])
        ij = int(np.flatnonzero(context["jitter_grid"] == 10)[0])
        oracle = np.mean([pupil.sampled_power_fraction * overlap_efficiency(propagate(
            pupil, 400 * make_ao_residual(pupil, 1.0, np.random.default_rng(610 + i), outer_scale_m=L0),
            cfg.ao.wavelength_um, t.pad_factor, tuple(10 * tilts[i])).field, mode) for i in range(frames)])
        assert context["modal_map"][iw, ij, 0] == pytest.approx(oracle, rel=1e-12)
        assert oracle == pytest.approx(expected[L0], abs=5e-7)
    near, far = (runs[L0] for L0 in (25.0, 0.5))
    # Zero WFE is the control: no screen, so the outer scale cannot matter.
    np.testing.assert_array_equal(near["modal_map"][0], far["modal_map"][0])
    assert not np.array_equal(near["modal_map"][1:], far["modal_map"][1:])
    for scan in ("wave_eta", "seeing_eta"):
        assert not np.array_equal(near[scan], far[scan])
