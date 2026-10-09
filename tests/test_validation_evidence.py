"""Notebook-execution evidence cannot outlive a failed or stale rerun."""

import importlib.util
import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path):
    """A minimal project tree holding a cleanly executed notebook and old evidence."""
    (tmp_path / "src/aophoton").mkdir(parents=True)
    (tmp_path / "src/aophoton/core.py").write_text("VALUE = 1\n")
    (tmp_path / "results").mkdir()
    (tmp_path / "notebooks").mkdir()
    cell = nbformat.v4.new_code_cell("1 + 1", execution_count=1)
    cell.outputs = [nbformat.v4.new_output("execute_result", {"text/plain": "2"}, execution_count=1)]
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), tmp_path / "notebooks/eso_astrophotonics_showcase.ipynb")
    (tmp_path / "results/notebook_execution.json").write_text(json.dumps(
        dict(status="passed", code_cells=12, error_outputs=0, png_figures=9, pdf_figures=9, wall_time_s=1.0)))
    return tmp_path


def test_failed_rerun_replaces_passing_record_and_blocks_doc_update(project, monkeypatch):
    execute = _load("execute_showcase")

    class FailingClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self):
            self.notebook.cells[0].outputs = [nbformat.v4.new_output(
                "error", ename="ValueError", evalue="boom", traceback=[])]
            raise RuntimeError("cell failed")

    class NoKernel:
        has_kernel = False

    monkeypatch.setattr(execute, "NotebookClient", FailingClient)
    monkeypatch.setattr(execute, "KernelManager", NoKernel)
    with pytest.raises(RuntimeError, match="cell failed"):
        execute.main(project)
    record = json.loads((project / "results/notebook_execution.json").read_text())
    assert record["status"] == "failed" and "cell failed" in record["error"]

    update = _load("update_result_sections")
    with pytest.raises(RuntimeError, match="did not pass"):
        update.main(project)


def test_passing_record_is_bound_to_notebook_and_source(project):
    evidence = _load("evidence")
    notebook = project / evidence.NOTEBOOK
    evidence.write_record(project, dict(
        status="passed", error_outputs=0, notebook_sha256=evidence.file_digest(notebook),
        source_sha256=evidence.source_digest(project)))
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


def test_pytest_evidence_is_bound_to_source_tests_and_junit(project):
    evidence = _load("evidence")
    (project / "tests").mkdir()
    (project / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    junit = project / evidence.JUNIT
    junit.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
    record = dict(status="passed", tests=1, source_sha256=evidence.source_digest(project),
                  tests_sha256=evidence.tests_digest(project), junit_sha256=evidence.file_digest(junit))
    (project / evidence.TEST_RECORD).write_text(json.dumps(record))
    assert evidence.verified_test_record(project)["tests"] == 1

    # A stale passing junit file cannot vouch for a changed test suite ...
    (project / "tests/test_x.py").write_text("def test_x():\n    assert False\n")
    with pytest.raises(RuntimeError, match="Tests or scripts changed"):
        evidence.verified_test_record(project)
    (project / "tests/test_x.py").write_text("def test_x():\n    pass\n")
    # ... nor can an edited pytest.xml or a failed run.
    junit.write_text('<testsuites><testsuite tests="9" failures="0" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(RuntimeError, match="pytest.xml does not match"):
        evidence.verified_test_record(project)
    (project / evidence.TEST_RECORD).write_text(json.dumps(dict(record, status="failed")))
    with pytest.raises(RuntimeError, match="did not pass"):
        evidence.verified_test_record(project)
