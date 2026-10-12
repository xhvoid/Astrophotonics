"""Validation evidence cannot outlive a failed or stale rerun, or vouch for
inputs, products or code other than those that actually produced it."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

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
    with pytest.raises(RuntimeError, match="cell failed"):
        execute.main(project)
    record = json.loads((project / "results/notebook_execution.json").read_text())
    assert record["status"] == "failed" and "cell failed" in record["error"]

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


@pytest.mark.parametrize("change", ["config", "metrics", "figure", "extra_product"])
def test_notebook_record_is_bound_to_inputs_and_products(project, change):
    # A passing record must not vouch for a different baseline configuration or
    # for result products edited (or added) after the run.
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
        (project / "figures/new.pdf").write_bytes(b"pdf")
        match = "Result products"
    with pytest.raises(RuntimeError, match=match):
        evidence.verified_record(project)


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
    record = dict(status="passed", tests=1, source_sha256=evidence.source_digest(project),
                  inputs_sha256=evidence.inputs_digest(project), imported_from=evidence.IMPORT_TARGET,
                  tests_sha256=evidence.tests_digest(project), junit_sha256=evidence.file_digest(junit))
    (project / evidence.TEST_RECORD).write_text(json.dumps(record))
    assert evidence.verified_test_record(project)["tests"] == 1

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


def test_complete_run_publishes_exactly_its_own_products(project, monkeypatch):
    execute = _load("execute_showcase")
    evidence = _load("evidence")
    (project / "figures/retired_plot.png").write_text("stale product of an earlier notebook\n")
    monkeypatch.setattr(execute, "NotebookClient", _staging_client(lambda stage: _write_products(stage, tag="new")))
    monkeypatch.setattr(execute, "KernelManager", _NoKernel)
    monkeypatch.setattr(execute, "_kernel_import_path", lambda kernel: str(project / "src/aophoton/__init__.py"))
    execute.main(project)
    assert not (project / "figures/retired_plot.png").exists()
    assert set(evidence.product_digests(project)) == set(evidence.REQUIRED_PRODUCTS)
    assert all((project / r).read_text().startswith("new") for r in evidence.REQUIRED_PRODUCTS)
    record = evidence.verified_record(project)
    assert record["status"] == "passed" and set(record["products_sha256"]) == set(evidence.REQUIRED_PRODUCTS)


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
    before = evidence.product_digests(project)
    calls, real_replace = [], os.replace

    def flaky_replace(source, target):
        calls.append(target)
        if len(calls) == 6:
            raise OSError("simulated write failure")
        return real_replace(source, target)

    monkeypatch.setattr(execute, "NotebookClient", _staging_client(lambda stage: _write_products(stage, tag="new")))
    monkeypatch.setattr(execute, "KernelManager", _NoKernel)
    monkeypatch.setattr(execute, "_kernel_import_path", lambda kernel: str(project / "src/aophoton/__init__.py"))
    monkeypatch.setattr(execute.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="simulated write failure"):
        execute.main(project)
    assert len(calls) > 6                                    # some files were replaced, then restored
    assert evidence.product_digests(project) == before
    assert json.loads((project / "results/notebook_execution.json").read_text())["status"] == "failed"
    assert not list(project.glob(".publish-backup-*")) and not list(project.rglob("*.publish-tmp"))


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
    # Storage that fails during publication AND restoration: the complete
    # previous set must survive in a backup that the error (and record) names.
    import os

    execute = _load("execute_showcase")
    evidence = _load("evidence")
    before = evidence.product_digests(project)
    calls, real_replace = [], os.replace

    def failing_storage(source, target):
        calls.append(target)
        if len(calls) > 3:
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
