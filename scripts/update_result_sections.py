"""Refresh documentation only from executed notebook and pytest evidence."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import verified_record, verified_test_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = '<!-- BEGIN GENERATED RESULTS -->', '<!-- END GENERATED RESULTS -->'

def replace_section(path: Path, body: str) -> None:
    text = path.read_text()
    before, remainder = text.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    path.write_text(before + BEGIN + '\n' + body.strip() + '\n' + END + after)


def main(root: Path = ROOT) -> None:
    # Only a passing record that still matches the current notebook and package
    # source is accepted; a failed or stale rerun refuses to refresh the docs.
    execution = verified_record(root)
    count = verified_test_record(root)['tests']
    metrics = json.loads((root/'results/showcase_metrics.json').read_text())
    b = metrics['baseline']
    table = f'''| Baseline quantity | Synthetic result |
|---|---:|
| Residual WFE | {b['residual_wfe_nm']:.0f} nm |
| Measured ensemble Strehl (with jitter) | {b['measured_strehl']:.4f} |
| Maréchal cross-check (without jitter) | {b['marechal_without_jitter']:.4f} |
| Mean SMF coupling across band | {b['mean_smf_coupling']:.4f} |
| Mean ideal six-mode acceptance | {b['mean_few_mode_coupling']:.4f} |
| Mean six-port lantern coupling, after 90% transmission | {b['mean_lantern_coupling']:.4f} |
| Median SNR per detector pixel | {b['median_snr_per_pixel']:.2f} |
| Injected RV | {b['injected_rv_m_s']:.2f} m/s |
| Recovered RV | {b['recovered_rv_m_s']:.2f} m/s |
| Recovery error | {b['recovery_bias_m_s']:+.2f} m/s |
| Conditional RV uncertainty (1σ) | {b['estimated_rv_uncertainty_m_s']:.2f} m/s |
| RV fit reduced χ² (≈1 when model and variance describe the data) | {b['rv_reduced_chi2']:.3f} |'''
    status = f"**Validation: {count} tests passed; all {execution['code_cells']} notebook code cells executed without errors; {execution['png_figures']} PNG/PDF figure pairs exported.** The last complete notebook execution took {execution['wall_time_s']:.1f} seconds on the validation machine (Python 3.14.3; CPU only). Runtime varies by machine."
    conclusions = '\n\n'.join(f'{i+1}. {text}' for i,text in enumerate(metrics['conclusions']))
    replace_section(root/'README.md',status+'\n\n'+table+'\n\n'+conclusions)
    replace_section(root/'docs/technical_note.md',status+'\n\n'+table+'\n\n'+conclusions+f'''

The design map uses {metrics['sweep_frames']} independent optical realizations per cell and common random inputs between alternatives. Its largest SMF standard error of the mean is {metrics['maximum_coupling_sem']:.3f}; uncertainty bars are shown in the pointing plot, and the map saves each standard error. Port-count rankings close to each other should not be interpreted as established optima. In particular, the 150 nm bright case is a nominal near tie between the SMF and 15-port model.

The no-jitter 150 nm illustrative PSF has measured Strehl {metrics['strehl_cases'][1][1]:.4f}, compared with Maréchal {metrics['strehl_cases'][1][2]:.4f}. At 400 nm the same wavefront shape gives {metrics['strehl_cases'][2][1]:.4f} versus {metrics['strehl_cases'][2][2]:.4f}, demonstrating why the approximation is not imposed at high aberration. The baseline instead averages independent PSFs and includes jitter, explaining its different measured Strehl.

Numbers come from [showcase_metrics.json](../results/showcase_metrics.json), [design_map.npz](../results/design_map.npz), [port_trade.json](../results/port_trade.json) and the [executed notebook](../notebooks/eso_astrophotonics_showcase.ipynb). Regenerate these artifacts before running `scripts/update_result_sections.py`.''')
    report = f'''# Validation report

This report is generated from saved pytest and notebook execution records. Validation is synthetic and numerical; no observed spectrum or optical-bench dataset has been fitted.

{status}

## Evidence

| Check | Evidence and scope |
|---|---|
| Package installation | Editable installation and distributable wheel; see `results/package_validation.json` for the separate-install verification. |
| End-to-end baseline | `results/showcase_baseline/summary.json`; configuration, provenance and package versions are retained. |
| Automated tests | `{count}` passed, zero failed/errors/skipped; [pytest.xml](../results/pytest.xml), bound with [pytest_record.json](../results/pytest_record.json) to the SHA-256 of the package source, the `configs/` inputs and the tests/scripts it ran, and to the `aophoton` module the session actually imported (this project's `src/`). |
| Fresh-kernel notebook | {execution['code_cells']} executed code cells, zero error outputs; [notebook_execution.json](../results/notebook_execution.json), bound to the SHA-256 of the executed notebook, package source, `configs/` inputs and every result product it wrote in `results/` and `figures/`, and to the `aophoton` module the kernel imported (run `{execution['run_id'][:12]}`). A failed or stale rerun, or an edited input or product, invalidates it and blocks this report. |
| Figures | {execution['png_figures']} plots, each in 300 dpi PNG and PDF, in `figures/`; all plots visually inspected for labels and layout. |
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
'''
    (root/'docs/validation_report.md').write_text(report)
    print(f'Refreshed documentation from {count} passing tests and the completed notebook.')

if __name__ == '__main__':
    main()
