"""Refresh documentation only from executed notebook and pytest evidence."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import unowned_files, verified_record, verified_test_record  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = '<!-- BEGIN GENERATED RESULTS -->', '<!-- END GENERATED RESULTS -->'

def replace_section(path: Path, body: str) -> None:
    text = path.read_text()
    before, remainder = text.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    path.write_text(before + BEGIN + '\n' + body.strip() + '\n' + END + after)


def closest_port_call(trade_rows: list[dict]) -> str:
    """Sentence naming the baseline-brightness WFE where SMF and the largest lantern rank closest."""
    bright = max(row['brightness_scale'] for row in trade_rows)
    most = max(row['ports'] for row in trade_rows)
    pairs = []
    for wfe in sorted({row['wfe_nm'] for row in trade_rows}):
        cases = {row['ports']: row['rv_sigma_m_s'] for row in trade_rows
                 if row['brightness_scale'] == bright and row['wfe_nm'] == wfe}
        if 1 in cases and most in cases:
            pairs.append((abs(cases[most] / cases[1] - 1), wfe, cases[1], cases[most]))
    gap, wfe, smf, lantern = min(pairs)
    return (f'In particular, at baseline brightness the SMF and {most}-port forecasts are closest at {wfe:g} nm WFE '
            f'({smf:.2f} versus {lantern:.2f} m/s, {gap:.1%} apart), a nominal near tie.')


def main(root: Path = ROOT) -> None:
    # Only a passing record that still matches the current notebook and package
    # source is accepted; a failed or stale rerun refuses to refresh the docs.
    execution = verified_record(root)
    count = verified_test_record(root)['tests']
    metrics = json.loads((root/'results/showcase_metrics.json').read_text())
    trade_rows = json.loads((root/'results/port_trade.json').read_text())
    b = metrics['baseline']
    if 'run_configuration' not in metrics:
        raise RuntimeError('showcase_metrics.json has no run_configuration; re-execute the notebook first.')
    # Every label below comes from the executed run, never from this script.
    run = metrics['run_configuration']
    fibre, spectrograph = run['fibre'], run['spectrograph']
    snr = b.get('median_snr_per_spectral_sample', b['median_snr_per_pixel'])
    pixels = b.get('detector_pixels_per_spectral_sample', run['detector']['extraction_pixels']
                   * (fibre['lantern_ports'] if fibre['feed'] == 'lantern' else 1))
    python = execution['python'].split()[0]
    context = (f"Baseline: {run['telescope']['diameter_m']:g} m pupil, {spectrograph['wavelength_min_um']:.3f}–"
               f"{spectrograph['wavelength_max_um']:.3f} µm band (AO reference {run['ao']['wavelength_um']:.3f} µm), "
               f"R = {spectrograph['resolving_power']:,.0f}, {fibre['feed']} feed, {b['noise_mode']} detector draw.")
    table = context + f'''

| Baseline quantity | Synthetic result |
|---|---:|
| Residual WFE | {b['residual_wfe_nm']:.0f} nm |
| Measured ensemble Strehl (with jitter) | {b['measured_strehl']:.4f} |
| Maréchal cross-check (without jitter) | {b['marechal_without_jitter']:.4f} |
| Mean SMF coupling across band | {b['mean_smf_coupling']:.4f} |
| Mean ideal {fibre['few_mode_count']}-mode acceptance | {b['mean_few_mode_coupling']:.4f} |
| Mean {fibre['lantern_ports']}-port lantern coupling, after {fibre['lantern_transmission']*100:g}% transmission | {b['mean_lantern_coupling']:.4f} |
| Median SNR per extracted spectral sample ({pixels} detector pixels) | {snr:.2f} |
| Injected RV | {b['injected_rv_m_s']:.2f} m/s |
| Recovered RV | {b['recovered_rv_m_s']:.2f} m/s |
| Recovery error | {b['recovery_bias_m_s']:+.2f} m/s |
| Conditional RV uncertainty (1σ) | {b['estimated_rv_uncertainty_m_s']:.2f} m/s |
| RV fit reduced χ² (≈1 when model and variance describe the data) | {b['rv_reduced_chi2']:.3f} |'''
    status = f"**Validation: {count} tests passed; all {execution['code_cells']} notebook code cells executed without errors; {execution['png_figures']} PNG/PDF figure pairs exported.** The last complete notebook execution took {execution['wall_time_s']:.1f} seconds on the validation machine (Python {python}; CPU only). Runtime varies by machine."
    conclusions = '\n\n'.join(f'{i+1}. {text}' for i,text in enumerate(metrics['conclusions']))
    replace_section(root/'README.md',status+'\n\n'+table+'\n\n'+conclusions)
    replace_section(root/'docs/technical_note.md',status+'\n\n'+table+'\n\n'+conclusions+f'''

The design map uses {metrics['sweep_frames']} independent optical realizations per cell and common random inputs between alternatives. Its largest SMF standard error of the mean is {metrics['maximum_coupling_sem']:.3f}; uncertainty bars are shown in the pointing plot, and the map saves each standard error. Port-count rankings close to each other should not be interpreted as established optima. {closest_port_call(trade_rows)}

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
| Automated tests | `{count}` passed, zero failed/errors/skipped; [pytest.xml](../results/pytest.xml), bound with [pytest_record.json](../results/pytest_record.json) to the SHA-256 of the package source, the `configs/` inputs, the tests/scripts it ran and `pyproject.toml`, and to the `aophoton` module the session actually imported (this project's `src/`). The record also binds the collection that `pyproject.toml` alone selects; a run narrowed by `-k`, `--deselect` or `PYTEST_ADDOPTS` is recorded as partial and certifies nothing. |
| Fresh-kernel notebook | {execution['code_cells']} executed code cells, zero error outputs; [notebook_execution.json](../results/notebook_execution.json), bound to the SHA-256 of the executed notebook, package source, `configs/` inputs and every result product it wrote in `results/` and `figures/`, and to the `aophoton` module the kernel imported (run `{execution['run_id'][:12]}`). The notebook runs in a fresh staging directory and its complete product set is published all-or-nothing (a failed publication restores the previous set). Each run records the products it owns in `results/notebook_products.json`. Only those, still unedited, are retired when a later run stops producing them; files no run wrote are never touched. A failed or stale rerun, or an edited input or product, invalidates it and blocks this report. |
| Figures | {execution['png_figures']} plots, each in 300 dpi PNG and PDF, in `figures/`; all plots visually inspected for labels and layout. |
| Repeated seeds | Pixel-for-pixel identical optical and noisy spectral arrays in the integration tests. |
| External validation | No OOPAO, observational, or manufactured-lantern validation performed. |

## Physical and numerical tolerances

- **Airy diffraction:** absolute profile error below 0.003 of peak over the first rings for a binary 128-pixel pupil; the first zero is within one 1/8 λ/D focal-plane sample.
- **FFT energy:** unit energy within 2×10⁻¹⁴ for multiple grid sizes, padding factors and obscured/spidered pupils. Odd-sized grids are included.
- **Atmospheric PSD:** the finite-grid ensemble variance is checked against represented frequency-bin power, with an analytic Gaussian quadratic-form sampling bound. Missing subharmonics are not mistaken for measured atmospheric variance.
- **High-Strehl cross-check:** absolute agreement with extended Maréchal within 0.004 at 15–60 nm WFE near 1.55 µm; the allowed fourth-order scale is documented in the test.
- **Complex coupling:** normalization, phase cancellation, [0,1] bounds, orthonormality, nested mode capacity, and a paired-ensemble degradation exceeding three standard errors under severe aberrations.
- **Spectral LSF/pixels:** Gaussian variance addition and the finite-pixel top-hat contribution are checked to 0.3%. A template at 0.5 over everything the model can reach (1.0 only beyond) returns exactly 0.5 in every pixel at every RV, with infinite Fisher width, and recovery reports no RV information (previously 8600.68 ± 1.66×10¹¹ m/s); altering only unreachable samples changes nothing, and a 0.2%-deep line on that plateau is still recovered. A feature inside the padding but >13 km/s outside the band at every searched RV adds no information at 30 km/s padding either (previously 3×10⁻¹¹ flux error, a −7204 m/s fit with finite sigma): each pixel whose truncated-LSF, projection and spline-coupling window holds one template value returns it exactly, the pixel width is integrated without cancellation, and the same feature is recovered once shifted into the band. `template_support_um` endpoints are accepted by the constructor for all 32 tested band/R pairs (previously 16 refused); a 1 m/s trim is refused. `highres_wavelength_um` is the fixed observer-frame grid (23,153 samples, 1.54–1.57 µm by default) at every RV. A 35 m/s line, seven times narrower than the internal grid step, keeps its analytic equivalent width to 10⁻⁸ at every sub-pixel and inter-node shift. Its centroid and width do not vary with the shift. The seeded template, generated by direct quadrature against the Gaussian-and-pixel response, is fitted with the default grid to within 0.05 m/s at R = 10,000 and 2×10⁻⁴ m/s at R = 100,000, with Fisher widths within 2×10⁻⁵.
- **Template import:** primary-array FITS wavelengths match `astropy.wcs` for linear `CD1_1` and `PC1_1 × CDELT1` axes, also beside a legal alternate description (`PC1_1A`, 1600–1608 nm next to a 1500–1504 nm primary). Non-linear, multi-axis, distorted or ambiguous primary headers are refused.
- **Detector statistics:** analytic source/background/dark/read budget and channel scaling, plus mean and variance from 100,000 independent samples.
- **Built-in template:** sampled at ≤ 200 m/s (3 per 600 m/s narrowest line sigma, at least 50,000 samples, so the default spectrum is unchanged). Against templates refined to 2–3× that density, a noiseless 2300 m/s injection across 1–2 µm at R = 30,000 is fitted within 0.1 m/s and 0.5% equivalent width (a fixed 50,000 samples gave 2329.93 m/s and +4.97%); the same holds at 1.45–1.80 µm and in the default band.
- **Zero-noise RV:** a non-grid injected velocity must be recovered within 0.01 m/s, approximately 10⁻⁵ of a detector pixel. This is a numerical same-operator check, not a claim of physical precision.
- **Wide RV search:** a noiseless injection recovered within 0.01 m/s over a ±400 km/s search; the coarse grid densifies to half the template's smallest velocity correlation scale, probed at least every half band-width of velocity across the search (so a line that enters even a 9-pixel band only away from the probes of a fixed set is still seen), and the three lowest local minima are refined. This is a heuristic, so a mismatched template or wrong minimum is flagged by reduced χ² far above its noise range. The correlation scale is the larger of the contrast/gradient and gradient/curvature ratios, so a symmetric two-pixel line searched over ±10 km/s takes under 300 model calls (previously a 2×10⁸-node grid). Every search is capped at 100,000 evaluations, checked before the grid is allocated; an unresolvable search fails with the scale and the work it would need.
- **Noisy RV uncertainty:** 64 independent recoveries must have standardized mean and variance within analytic three-sigma ensemble bounds. This supports local conditional uncertainty only; it does not validate calibration floors.
- **Pupil geometry:** a 1 mm crossed spider on a clear 3.6 m pupil removes its exact 0.0072 m² to within one pixel of chord sampling (previously 0.77 m² at 32 pixels). Its coupling change converges to a continuous pupil-plane overlap at O(pixel/diameter).
- **Pointing:** net tip-tilt (the OPD's least-squares plane over the illuminated pupil plus explicit pointing) beyond half the periodic focal field's half-width is refused, rather than wrapped to the opposite sign; equal and opposite contributions reproduce the flat PSF, and both encodings give one PSF where the field is wide enough.
- **Notebook sweeps:** the design map, wavelength and seeing scans use the configured outer scale; at 0.5 m the 400 nm / 10 mas SMF cell matches a same-seed explicit calculation (0.051988, against 0.050513 at 25 m).
- **Mode bases:** orthonormality is checked in float64/complex128, so boolean or small-integer samples cannot overflow into a false pass.
- **Run records:** CLI summaries record the noise mode, template content digest and a run identity. Each mode replays from its own summary, and changed template bytes are refused. Replay compares 19 scientific results (every feed's coupling, the boundary flag, derived RV quantities) before writing anything; a changed field is named, and the reference and any existing destination, including the reference's own directory through an alias, keep their bytes.
- **Publication:** concurrent publishers commit whole generations under a project lock, and a failing publisher restores the generation committed before it, never an older one. The passing record is the commit: a fault injected into its rename, after another run's generation had been replaced, restores that generation's notebook, manifest and all 26 products byte for byte (previously none survived). An older run whose cell fails after a newer run committed leaves the newer notebook in place and keeps its partial notebook under its own run ID (previously it replaced the published notebook). Temporary files are created exclusively, so a file or link at a predictable name, including a link at the execution record for each of its three writes or at the pytest record and `pytest.xml` (even one made during the test session), is never written through. Test runs of one project take turns under their own lock, and each record's counts and digest come from the report its own session wrote: with one run held just after publishing `pytest.xml`, a second run now waits for it, and both records describe their own XML (previously the second run's XML replaced the first's, and the first run's record certified it). Unrelated files in the product folders are listed but never opened (checked with an audit hook).
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
    foreign = unowned_files(root, execution)
    if foreign:
        print('Not part of the notebook evidence (left untouched): ' + ', '.join(foreign))

if __name__ == '__main__':
    main()
