# Validation report

This report is generated from saved pytest and notebook execution records. Validation is synthetic and numerical; no observed spectrum or optical-bench dataset has been fitted.

**Validation: 76 tests passed; all 12 notebook code cells executed without errors; 9 PNG/PDF figure pairs exported.** The last complete notebook execution took 19.9 seconds on the validation machine (Python 3.14.3; CPU only). Runtime varies by machine.

## Evidence

| Check | Evidence and scope |
|---|---|
| Package installation | Editable installation and distributable wheel; see `results/package_validation.json` for the separate-install verification. |
| End-to-end baseline | `results/showcase_baseline/summary.json`; configuration, provenance and package versions are retained. |
| Automated tests | `76` passed, zero failed/errors/skipped; [pytest.xml](../results/pytest.xml), bound with [pytest_record.json](../results/pytest_record.json) to the SHA-256 of the package source, the `configs/` inputs and the tests/scripts it ran, and to the `aophoton` module the session actually imported (this project's `src/`). |
| Fresh-kernel notebook | 12 executed code cells, zero error outputs; [notebook_execution.json](../results/notebook_execution.json), bound to the SHA-256 of the executed notebook, package source, `configs/` inputs and every result product it wrote in `results/` and `figures/`, and to the `aophoton` module the kernel imported (run `e12dab318ec5`). The notebook runs in a fresh staging directory and its complete product set is published all-or-nothing (a failed publication restores the previous set). A failed or stale rerun, or an edited input or product, invalidates it and blocks this report. |
| Figures | 9 plots, each in 300 dpi PNG and PDF, in `figures/`; all plots visually inspected for labels and layout. |
| Repeated seeds | Pixel-for-pixel identical optical and noisy spectral arrays in the integration tests. |
| External validation | No OOPAO, observational, or manufactured-lantern validation performed. |

## Physical and numerical tolerances

- **Airy diffraction:** absolute profile error below 0.003 of peak over the first rings for a binary 128-pixel pupil; the first zero is within one 1/8 λ/D focal-plane sample.
- **FFT energy:** unit energy within 2×10⁻¹⁴ for multiple grid sizes, padding factors and obscured/spidered pupils. Odd-sized grids are included.
- **Atmospheric PSD:** the finite-grid ensemble variance is checked against represented frequency-bin power, with an analytic Gaussian quadratic-form sampling bound. Missing subharmonics are not mistaken for measured atmospheric variance.
- **High-Strehl cross-check:** absolute agreement with extended Maréchal within 0.004 at 15–60 nm WFE near 1.55 µm; the allowed fourth-order scale is documented in the test.
- **Complex coupling:** normalization, phase cancellation, [0,1] bounds, orthonormality, nested mode capacity, and a paired-ensemble degradation exceeding three standard errors under severe aberrations.
- **Spectral LSF/pixels:** Gaussian variance addition and the finite-pixel top-hat contribution are checked to 0.3%; equivalent width is conserved to the stated quadrature tolerance.
- **Detector statistics:** analytic source/background/dark/read budget and channel scaling, plus mean and variance from 100,000 independent samples.
- **Zero-noise RV:** a non-grid injected velocity must be recovered within 0.01 m/s, approximately 10⁻⁵ of a detector pixel. This is a numerical same-operator check, not a claim of physical precision.
- **Wide RV search:** a noiseless injection recovered within 0.01 m/s over a ±400 km/s search; the coarse grid densifies to half the template's smallest velocity correlation scale, probed at least every half band-width of velocity across the search (so a line that enters even a 9-pixel band only away from the probes of a fixed set is still seen), and the three lowest local minima are refined. This is a heuristic, so a mismatched template or wrong minimum is flagged by reduced χ² far above its noise range.
- **Noisy RV uncertainty:** 64 independent recoveries must have standardized mean and variance within analytic three-sigma ensemble bounds. This supports local conditional uncertainty only; it does not validate calibration floors.
- **Input handling:** unknown configuration keys, invalid units/arrays, insufficient template coverage (including the 1 m/s Fisher-derivative margin beyond the search), fibre modes unresolved by the focal-plane sampling or truncated by the grid, absent RV information and truncated searches are explicitly checked or diagnosed.
- **Modal sampling:** the focal-plane sampling floor grows with Hermite-Gaussian degree n as 1.1 + 0.31·√(2n+1) pixels per 1/e radius (1.41 px for a single mode, 2.38 px at degree 8), calibrated for degrees 0–8 so the acceptance of fields displaced by up to five radii and aberrated by up to 400 nm agrees with a padding-16 reference to better than 0.1% (higher degrees use the same floor as an extrapolation; independent checks at degrees 16 and 24 found 0.11–0.15%); a 45-mode basis previously accepted at 1.30 px overestimated acceptance by 62%.

## Reproducing the report

```bash
python scripts/run_tests.py
python scripts/execute_showcase.py
python scripts/update_result_sections.py
```

See [technical_note.md](technical_note.md) for equations, assumptions and interpretation, and [public_data.md](public_data.md) for a future observational procedure. The model omits finite-exposure coupling fluctuations, AO temporal dynamics, real fibre/lantern behaviour, tellurics, stellar variability and instrument-calibration systematics.
