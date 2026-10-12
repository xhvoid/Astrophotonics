# AO-Photonic Spectrograph Performance Simulator

**How do seeing, residual adaptive-optics wavefront error, tip-tilt jitter, and photonic-lantern mode count change fibre coupling, spectral SNR, and radial-velocity precision?**

A small, reproducible Python project connecting complex wave optics to a synthetic spectroscopic measurement. It is a **NIRPS-like, ANDES-inspired conceptual trade study** for astronomical-instrumentation and astrophotonics portfolios.

> This is an independent conceptual study and is not an official ESO instrument model.

[**Open the executed showcase notebook**](notebooks/eso_astrophotonics_showcase.ipynb) · [Technical note](docs/technical_note.md) · [Validation report](docs/validation_report.md) · [References](docs/references.md)

![Residual WFE and pointing jitter determine coupling; modal acceptance changes the trade.](figures/core_design_map.png)

The core map averages 12 independent realizations per cell at 1.555 µm. The lantern curves assume a 90% transition transmission and an ideal orthonormal modal subspace. Figures are available as [300 dpi PNG](figures/core_design_map.png) and [vector PDF](figures/core_design_map.pdf); uncertainty and detector-cost comparisons appear in the notebook.

## Why this matters for ESO instrumentation

Coupling an AO-corrected focal plane into a photonic input connects optical alignment, wavefront quality, throughput and detector performance. [ESO's NIRPS description](https://www.eso.org/sci/facilities/develop/instruments/NIRPS.html) supplies the 3.6 m aperture, historical HAM resolving-power baseline of 100,000, and three-pixel sampling used as design inspiration. These are historical design choices, not a reconstruction of installed NIRPS performance. [ANDES](https://arxiv.org/abs/2407.14601) motivates the broader ELT-era high-resolution NIR spectroscopy question; this code implements a circular 3.6 m pupil.

The project demonstrates physical modelling, unit-aware interfaces, reproducible experiments, numerical validation and communication of engineering trade-offs. A proposed [photonic-chip testbench comparison](https://www.eso.org/sci/activities/fellowships-and-studentships/engstudentfellow/photonic_chips.html) provides a practical next step.

## Install and run

Python **3.10 or newer**; CPU only. This is a standalone repository. Enter the project directory; the example below uses `~/Documents/Astrophotonics`.

```bash
cd ~/Documents/Astrophotonics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
aophoton simulate --config configs/baseline.yaml
python -m pytest -q
python scripts/execute_showcase.py
```

The CLI writes `results/baseline/summary.json`, `spectrum.csv`, and `arrays.npz`. The JSON includes the full configuration, every setting's provenance, package versions, Strehl/coupling/SNR and RV results. The CSV includes expected and noisy electrons, their variance, SNR, and wavelength-dependent coupling. The notebook executor uses the launching Python interpreter, starts a fresh kernel, rejects cell errors, saves outputs, verifies figure pairs and closes the kernel.

For just the simulation, use `pip install -e .`; FITS support is optional through `.[fits]`. `.[dev]` includes tests, FITS and notebook execution. No downloads are performed by the simulation or showcase. Interactive notebook frontends are optional; the execution script is sufficient to reproduce the saved notebook.

```bash
# Synthetic zero-noise validation, with a separate output directory:
aophoton simulate --config configs/baseline.yaml --no-noise --output results/noiseless

# After editing notebook source cells in the optional builder:
python scripts/build_showcase.py
python scripts/execute_showcase.py
```

Edit [baseline.yaml](configs/baseline.yaml) to change seeing, WFE, jitter, wavelength, resolution, source brightness or detector terms. Set `fibre.feed` to `smf`, `few_mode`, or `lantern`; configure `lantern_ports` for its spatial capacity. Unknown configuration keys fail explicitly. Imported template paths are relative to the YAML file; formats and observational caveats are described in [public_data.md](docs/public_data.md).

## Reproduced results

<!-- BEGIN GENERATED RESULTS -->
**Validation: 70 tests passed; all 12 notebook code cells executed without errors; 9 PNG/PDF figure pairs exported.** The last complete notebook execution took 17.1 seconds on the validation machine (Python 3.14.3; CPU only). Runtime varies by machine.

| Baseline quantity | Synthetic result |
|---|---:|
| Residual WFE | 150 nm |
| Measured ensemble Strehl (with jitter) | 0.6598 |
| Maréchal cross-check (without jitter) | 0.6926 |
| Mean SMF coupling across band | 0.4711 |
| Mean ideal six-mode acceptance | 0.5303 |
| Mean six-port lantern coupling, after 90% transmission | 0.4772 |
| Median SNR per detector pixel | 144.06 |
| Injected RV | 2300.00 m/s |
| Recovered RV | 2296.37 m/s |
| Recovery error | -3.63 m/s |
| Conditional RV uncertainty (1σ) | 4.12 m/s |
| RV fit reduced χ² (≈1 when model and variance describe the data) | 0.962 |

1. At zero jitter, increasing residual WFE from 0 to 500 nm changes mean SMF coupling from 0.699 to 0.020; wavefront quality sets the photon budget.

2. At 150 nm WFE, increasing per-axis jitter from 0 to 40 mas changes SMF coupling from 0.489 to 0.249; injection stability remains important even at fixed AO residual.

3. At 400 nm WFE / 10 mas jitter, replacing the SMF with a 15-port lantern changes forecast σRV from 12.98 to 8.75 m/s at baseline brightness, but from 403.84 to 502.86 m/s at 1/100 brightness. Under the assumed insertion loss and extraction noise, detector costs can reverse the throughput advantage.
<!-- END GENERATED RESULTS -->

These RV errors are conditional statistical forecasts with an exact template and calibration. They are not predicted NIRPS or ANDES accuracy. A single noisy recovery error does not establish an estimator's ensemble bias.

## Model and scope

```text
Atmosphere → AO residual → complex focal field / PSF → fibre / lantern
           → Gaussian spectrograph LSF → detector electrons → RV fit
```

- **Optics:** circular pupil, configurable obscuration and simplified spiders, reproducible von Kármán screens, scalar residual-RMS AO prescription, pointing offsets, unitary Fraunhofer FFT, monochromatic and incoherent polychromatic PSFs, measured Strehl and extended Maréchal cross-check.
- **Photonic input:** normalized complex overlap with a Gaussian SMF; nested orthonormal Hermite–Gaussian modes approximate ideal few-mode acceptance. Lanterns transmit the accepted subspace with an assumed scalar loss. There is no arbitrary random transfer matrix.
- **Spectroscopy:** seeded synthetic absorption lines or explicit ASCII/FITS template, relativistic Doppler injection, constant-R Gaussian LSF, finite pixel integration, staged throughput, Poisson source/background/dark counts and independent read noise.
- **RV:** coarse search and continuous χ² template matching; the coarse grid is densified to half the template's smallest velocity correlation scale, probed at least every half band-width of velocity across the search so wide searches do not step over the true basin, and the lowest local minima are each refined (a heuristic: reduced χ² flags a wrong minimum); local spectral-gradient Fisher uncertainty; reduced χ², zero-information and search-boundary diagnostics. Independent lantern outputs add extraction noise while conserving source photons.

All interface units appear in names/docstrings: metres, micrometres, nanometres, arcseconds, milliarcseconds, seconds, electrons and m/s. The complete baseline parameter table labels **sourced**, **assumed** and **illustrative** settings. There are **no fitted instrument parameters**. A sourced default that is changed is relabelled as assumed.

Key assumptions are deliberately visible:

1. `residual_wfe_nm` denotes pupil RMS **at the reference seeing**. At fixed assumed AO attenuation, actual WFE scales as `(seeing/reference_seeing)^(5/6)`. At fixed actual WFE, seeing has no separate independent penalty in this model. Screens are periodic, lack subharmonics, and have exactly imposed RMS; they do not simulate an AO controller.
2. Jitter is Gaussian RMS **per axis**; independent frames have no physical cadence. Intensities and coupling powers are averaged, not complex fields. Finite-exposure throughput fluctuations are omitted from spectral variance.
3. The Gaussian radius is an angular 1/e field radius, representing the injection relay. HG modes approximate a parabolic-index guide; port counts are scalar spatial capacities with polarization effects omitted. No physical taper, manufacturing model, modal noise or real fibre cutoff is inferred.
4. The spectrum has an exact continuum/template and wavelength solution. It omits tellurics, stellar variability, instrumental drift, correlated detector noise, saturation, echelle geometry and extraction cross-talk.

An [OOPAOAdapter](src/aophoton/adapters.py) accepts explicitly matched external OPD snapshots. **OOPAO integration and observational validation remain future work.** No full ELT control system, ray tracing, Maxwell/FDTD propagation, neural network, GUI, or proprietary data is included.

## Package structure

| Responsibility | Modules / artifacts |
|---|---|
| Configuration, provenance, CLI | `config.py`, `cli.py`, `configs/baseline.yaml` |
| Pupil, phase screens, PSFs | `pupil.py`, `atmosphere.py`, `propagation.py`, `metrics.py` |
| Photonic coupling | `coupling.py`, `lantern.py` |
| Spectrum, LSF, electrons, RV | `spectrum.py`, `spectrograph.py`, `detector.py`, `radial_velocity.py` |
| End-to-end simulation / extension | `simulation.py`, `adapters.py` |
| Scientific validation | `tests/`, `docs/validation_report.md` |
| Application-facing story | executed notebook, `figures/`, `results/` |

The surrounding AO-fed IFU project uses intensity PSF proxies and spaxel coupling. This package needs complex focal fields and coherent modal overlaps, so it has its own `src/aophoton` implementation and does not import or modify those modules.

## Reproducibility and contribution

The saved notebook contains the sweep grids, seeds, numerical experiments, figures and computed conclusions. [showcase_metrics.json](results/showcase_metrics.json), [design_map.npz](results/design_map.npz) and [port_trade.json](results/port_trade.json) retain its numerical outputs. [requirements-validated.txt](requirements-validated.txt) records the validated environment; `pyproject.toml` specifies broader supported dependencies. Last-digit equality across different FFT/BLAS versions is not guaranteed.

To refresh the evidence after a change:

```bash
python scripts/run_tests.py
python scripts/execute_showcase.py
python scripts/update_result_sections.py
```

Core tests compare analytic diffraction, energy conservation, small-error theory, physical bounds and statistical ensembles; they do not merely compare to stored reference outputs. Improvements should add a physically justified validation and update the technical note. Large observational files belong in the ignored `data/` directory, with provenance recorded separately.

[MIT license](LICENSE). Primary equation, instrument and software sources are collected in [references.md](docs/references.md).
