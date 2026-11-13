# Reproducibility

The generated results sections of the README and the [technical note](technical_note.md), and the whole [validation report](validation_report.md), are written from saved test and notebook runs. This page describes how to regenerate those runs and how the saved records are checked against the current tree.

## Regenerating the evidence

From an environment installed with `python -m pip install -e '.[dev]'`:

```bash
python scripts/run_tests.py
python scripts/execute_showcase.py
python scripts/update_result_sections.py
```

1. `run_tests.py` runs pytest with a JUnit report and writes [pytest_record.json](../results/pytest_record.json). The previous record is invalidated before the run, so a failed or interrupted run cannot leave an older passing record in place. It first collects the suite that `pyproject.toml` alone selects, with `PYTEST_ADDOPTS`, `PYTEST_PLUGINS` and `addopts` removed, then runs the tests as the caller configured them. The record passes only if exactly that collection ran and every test passed. A narrower selection (`-k`, `--deselect`, `--lf`) that passes is recorded as `partial` and blocks the documentation update. The test count is whatever the configuration collects. Neither the record nor `pytest.xml` is written in place: pytest writes its JUnit report into the run's private scratch folder, and both files are renamed into place from fresh temporary files, so a symbolic link at either name, even one made while the tests run, is replaced rather than written through. The record's counts and JUnit digest are taken from the bytes of that private report, which are the bytes published, so a run without a readable report of its own fails whatever file is at the public path. Test runs of one project take turns: a run holds an exclusive lock on `results/.pytest_run.lock` (POSIX `flock`, ignored by Git) from before its `running` record and the removal of the old `pytest.xml` until its final record is written, and a second run waits, saying so, until it is released. The lock covers the whole run rather than only the final two writes, so that another run's `running` record or report removal can never fall between them and an earlier run's final record can never replace a later run's `running` one. Runs of different checkouts have different locks and never wait for each other.
2. `execute_showcase.py` executes the [showcase notebook](../notebooks/eso_astrophotonics_showcase.ipynb) in a fresh kernel and writes [notebook_execution.json](../results/notebook_execution.json).
3. `update_result_sections.py` refreshes the generated sections of the README and technical note and rewrites the validation report. It refuses to run if either record is failed or stale.

Interactive notebook frontends are optional; the execution script is sufficient to reproduce the saved notebook.

## Notebook execution and publication

The notebook executor uses the launching Python interpreter and runs in a fresh staging directory. It requires all 25 expected result and figure files before publishing. Read-only destinations are rejected before changes; other publication errors trigger hash-verified restoration of the previous set. If storage errors prevent restoration, the error identifies the retained backup.

Publication is one transaction per project. The ownership read, the products, the executed notebook, the manifest, the execution record, verification and any rollback all run under an exclusive lock on `results/.publication.lock` (POSIX `flock`, released by the kernel if its holder dies). Two runs started together therefore commit one after the other, and the later one retires what the earlier one's manifest owns. The passing execution record is the commit: it is renamed into place last, after the published set has been verified, and the backup of the previous generation is kept until that rename has succeeded. A failure at any step up to and including it restores the generation that was committed when the run took the lock, which may be another run's, and the lock is held until the failure has been recorded. A failed *execution* never publishes and never touches the published notebook, which belongs to the installed products; its partial notebook is kept as `results/failed_executions/<run_id>.ipynb` (ignored by Git) and named in the failed record. Every published file, including the execution record at each of its `running`, `passed` and `failed` writes, is written to a fresh, exclusively created temporary file in its destination folder and renamed into place, so an existing file or symbolic link at any name is never written through, and a failed write leaves the previous file as it was. Files in the product folders that no run owns are listed by name only; their contents are never read, so their size or permissions cannot affect a publication.

Each published run writes [notebook_products.json](../results/notebook_products.json), the products it owns and their SHA-256. A later run removes a product it no longer writes only if that manifest owns it and the file is unchanged since. Ownership is never inferred:

- A user file in `results/` or `figures/`, such as a bench `figures/alignment_scan.png`, is never removed or certified.
- An owned product that was edited afterwards is kept.
- A tree from before the manifest falls back to the passing execution record's product list, and otherwise retires nothing.
- A corrupt manifest also retires nothing.

A run whose output would overwrite a file no run wrote is refused before anything changes. The updater lists, without reading, files in those folders that are not part of the evidence.

## What a passing record binds

| Record | Bound to (SHA-256) |
|---|---|
| `results/pytest_record.json` | Package source, `configs/` inputs and referenced templates, `tests/` and `scripts/`, `pyproject.toml`, the intended test collection, the JUnit report that run's own session wrote, and the `aophoton` module the session imported |
| `results/notebook_execution.json` | Executed notebook, package source, `configs/` inputs and referenced templates, every result and figure the run published (and nothing else in those folders), and the `aophoton` module the kernel imported |
| `results/package_validation.json` | Records the built wheel after installing it outside the checkout and running its CLI; not checked by the documentation updater |

`update_result_sections.py` re-checks the first two records against the current tree. Editing any file they bind makes them stale until they are regenerated; README prose outside the generated sections is not bound.

## Editing the notebook

The notebook source cells are generated by an optional builder. After changing it:

```bash
python scripts/build_showcase.py
python scripts/execute_showcase.py
```

## Wheel verification

```bash
python -m pip wheel --no-deps . -w dist
python scripts/verify_wheel.py
```

`dist/` must contain exactly one wheel. The README is the package's long description, so a README change also changes the wheel.
