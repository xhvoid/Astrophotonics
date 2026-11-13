"""Seeded end-to-end orchestration, with independent optical and detector streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from .atmosphere import make_ao_residual
from .config import SimulationConfig, parameter_records
from .coupling import gaussian_mode, overlap_efficiency
from .detector import DetectorResult, simulate_detector
from .lantern import mode_basis, modal_coupling
from .metrics import marechal_strehl, rms_wfe_nm, strehl_ratio
from .propagation import FocalPlane, propagate
from .pupil import make_pupil
from .radial_velocity import RVResult, recover_rv
from .spectrograph import SpectrographConfig, SpectrographModel, template_support_um
from .spectrum import Spectrum, load_spectrum, synthetic_spectrum


# Margin [um] of the built-in template beyond the band. Kept whenever it
# already covers the model's support, so the seeded default spectrum is
# unchanged; low resolving power or wide RV padding extends it.
SYNTHETIC_TEMPLATE_MARGIN_UM = 0.005
# Relative widening (0.3 m/s) of a support-derived limit, so that rounding of
# the template's log-spaced end samples cannot fail the coverage check.
SUPPORT_ROUNDING_GUARD = 1e-9


def synthetic_template_range_um(spectrograph: SpectrographConfig) -> tuple[float, float]:
    """Wavelength range [um] of the built-in template for ``spectrograph``.

    The band plus ``SYNTHETIC_TEMPLATE_MARGIN_UM``, widened where needed to the
    support the model requires: the band, its Doppler padding and the
    truncated LSF (``template_support_um``).
    """
    low, high = template_support_um(spectrograph)
    return (min(spectrograph.wavelength_min_um - SYNTHETIC_TEMPLATE_MARGIN_UM, low * (1 - SUPPORT_ROUNDING_GUARD)),
            max(spectrograph.wavelength_max_um + SYNTHETIC_TEMPLATE_MARGIN_UM, high * (1 + SUPPORT_ROUNDING_GUARD)))


def simulation_template(config: SimulationConfig) -> Spectrum:
    """The configured imported template, or the seeded built-in one covering the model."""
    return _template_with_provenance(config)[0]


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def spectrum_sha256(spectrum: Spectrum) -> str:
    """SHA-256 of a template's float64 little-endian wavelength then flux samples."""
    digest = hashlib.sha256()
    for values in (spectrum.wavelength_um, spectrum.relative_flux):
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _template_with_provenance(config: SimulationConfig) -> tuple[Spectrum, dict[str, Any]]:
    """Template and the record that identifies it: file bytes, or generator inputs."""
    o = config.observation
    if o.template_path:
        before = file_sha256(o.template_path)
        template = load_spectrum(o.template_path)
        if file_sha256(o.template_path) != before:
            raise ValueError(f"Template {o.template_path} changed while it was read.")
        record = dict(source="file", path=str(o.template_path), file_sha256=before,
                      file_bytes=Path(o.template_path).stat().st_size)
    else:
        low, high = synthetic_template_range_um(config.spectrograph)
        template = synthetic_spectrum(low, high, seed=o.template_seed)
        record = dict(source="synthetic", generator="aophoton.spectrum.synthetic_spectrum",
                      wavelength_min_um=low, wavelength_max_um=high, seed=o.template_seed)
    record.update(samples=int(template.wavelength_um.size), samples_sha256=spectrum_sha256(template))
    return template, record


@dataclass
class OpticalResult:
    """Ensemble mean efficiencies and PSF; wavelength in µm, angles in mas."""

    wavelengths_um: np.ndarray
    smf_efficiency: np.ndarray
    few_mode_efficiency: np.ndarray
    lantern_efficiency: np.ndarray
    modal_power: np.ndarray
    mean_psf: np.ndarray
    reference: FocalPlane
    first_frame: FocalPlane
    first_opd_m: np.ndarray
    strehl: float
    marechal: float
    residual_wfe_nm: float
    collecting_area_m2: float


@dataclass
class SimulationResult:
    """Optics, sampled expected/noisy spectrum, and a fitted radial velocity."""

    config: SimulationConfig
    optics: OpticalResult
    spectral_model: SpectrographModel
    detector: DetectorResult
    rv: RVResult
    coupling: np.ndarray
    noisy: bool = True
    template: dict[str, Any] = field(default_factory=dict)

    @property
    def detector_pixels_per_spectral_sample(self) -> int:
        """Physical pixels summed into one extracted spectral sample."""
        f = self.config.fibre
        return self.config.detector.extraction_pixels * (f.lantern_ports if f.feed == "lantern" else 1)

    @property
    def observed_column(self) -> str:
        """CSV name of the background-subtracted signal, by noise mode."""
        return "background_subtracted_noisy_e" if self.noisy else "background_subtracted_expected_e"

    def snr_definition(self) -> str:
        f, n = self.config.fibre, self.config.detector.extraction_pixels
        channels = f.lantern_ports if f.feed == "lantern" else 1
        return (f"Expected source electrons / sqrt(forecast variance) per extracted spectral sample: one "
                f"detector-pixel-wide wavelength bin summed over {n} spatial extraction pixel(s) in each of "
                f"{channels} independently read channel(s), {self.detector_pixels_per_spectral_sample} physical "
                "pixels in all. Not per physical detector pixel.")

    def csv_columns(self) -> dict[str, str]:
        """Exported spectrum.csv columns, in order, with units and meaning."""
        observed = ("Background-subtracted electrons of the single seeded noisy draw (Poisson source, dark "
                    "and background; Gaussian read noise)" if self.noisy else
                    "Background-subtracted electrons without noise: equal to expected_source_e (noiseless mode)")
        return {
            "wavelength_um": "Geometric-centre wavelength of the detector pixel [um]",
            "expected_source_e": "Expected source electrons per extracted spectral sample",
            self.observed_column: observed,
            "variance_e2": "Forecast variance per extracted spectral sample [e^2]: source, dark, background "
                           "and read noise of every summed pixel",
            "snr_per_spectral_sample": self.snr_definition(),
            "coupling": "Feed coupling efficiency, interpolated from the optical wavelength nodes",
        }

    def run_identity(self) -> str:
        """SHA-256 of what determines this run: configuration, noise mode and template content.

        Template file paths are left out, so the identity follows the bytes,
        not where they are stored.
        """
        configuration = asdict(self.config)
        configuration["observation"].pop("template_path")
        template = {key: value for key, value in self.template.items() if key != "path"}
        identity = dict(configuration=configuration, noise_mode=self.noise_mode, template=template)
        return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()

    @property
    def noise_mode(self) -> str:
        return "noisy" if self.noisy else "noiseless"

    def summary(self) -> dict[str, Any]:
        """JSON-compatible scalar results; single-realization error is not an ensemble bias."""
        snr = float(np.median(self.detector.snr))
        return dict(
            disclaimer="This is an independent conceptual study and is not an official ESO instrument model.",
            run_identity_sha256=self.run_identity(), noise_mode=self.noise_mode,
            seed=self.config.seed, feed=self.config.fibre.feed, template=self.template,
            residual_wfe_nm=self.optics.residual_wfe_nm,
            measured_strehl=self.optics.strehl,
            marechal_without_jitter=self.optics.marechal,
            mean_smf_coupling=float(np.mean(self.optics.smf_efficiency)),
            mean_few_mode_coupling=float(np.mean(self.optics.few_mode_efficiency)),
            mean_lantern_coupling=float(np.mean(self.optics.lantern_efficiency)),
            mean_selected_coupling=float(np.mean(self.coupling)),
            collecting_area_m2=self.optics.collecting_area_m2,
            median_snr_per_spectral_sample=snr,
            detector_pixels_per_spectral_sample=self.detector_pixels_per_spectral_sample,
            snr_definition=self.snr_definition(),
            median_snr_per_pixel=snr,
            injected_rv_m_s=self.config.observation.injected_rv_m_s,
            recovered_rv_m_s=self.rv.rv_m_s,
            recovery_bias_m_s=self.rv.rv_m_s-self.config.observation.injected_rv_m_s,
            estimated_rv_uncertainty_m_s=self.rv.uncertainty_m_s,
            rv_at_search_boundary=bool(self.rv.at_boundary),
            rv_reduced_chi2=float(self.rv.reduced_chi2),
            rv_model_evaluations=int(self.rv.model_evaluations),
            validation="Synthetic only; no OOPAO or observational validation",
            field_descriptions={
                "noise_mode": "noisy: one seeded Poisson/read-noise draw; noiseless: expected counts with the "
                              "same forecast variance (injection/recovery check)",
                "run_identity_sha256": "SHA-256 of the configuration, noise mode and template record; "
                                       "replayable with `aophoton replay`",
                "template": "Template source; a file is identified by the SHA-256 of its bytes, a synthetic "
                            "template by its generator inputs; samples_sha256 covers the samples used",
                "median_snr_per_spectral_sample": "Median over the band of snr_per_spectral_sample (see "
                                                  "snr_definition)",
                "median_snr_per_pixel": "Deprecated alias of median_snr_per_spectral_sample, kept for "
                                        "compatibility; it is not per physical detector pixel",
                "estimated_rv_uncertainty_m_s": "Fisher 1-sigma forecast from the expected spectrum [m/s]",
                "recovery_bias_m_s": "Recovered minus injected RV of this single realization [m/s]; "
                                     "not an ensemble bias",
                "collecting_area_m2": "Geometric clear pupil area [m^2]",
                "spectrum_csv_columns": "Columns of spectrum.csv, in order",
            },
            spectrum_csv_columns=self.csv_columns(),
        )


def simulate_optics(config: SimulationConfig) -> OpticalResult:
    """Average independent short-exposure intensities/powers, never complex fields."""
    t, a, f = config.telescope, config.ao, config.fibre
    pupil = make_pupil(t.n_pix, t.diameter_m, t.obscuration, t.spider_width_m)
    wavelengths = np.linspace(config.spectrograph.wavelength_min_um,
                              config.spectrograph.wavelength_max_um, a.wavelength_nodes)
    n_modes = max(f.few_mode_count, f.lantern_ports)
    references = [propagate(pupil, np.zeros_like(pupil.amplitude), w, t.pad_factor) for w in wavelengths]
    bases = [mode_basis(ref, f.mode_radius_mas, n_modes) for ref in references]
    gaussians = [gaussian_mode(ref, f.mode_radius_mas) for ref in references]
    reference = propagate(pupil, np.zeros_like(pupil.amplitude), a.wavelength_um, t.pad_factor)
    mean_psf = np.zeros_like(reference.intensity)
    powers = np.zeros((len(wavelengths), n_modes))
    smf = np.zeros(len(wavelengths))
    phase_seed, jitter_seed, _ = np.random.SeedSequence(config.seed).spawn(3)
    phase_rng, jitter_rng = np.random.default_rng(phase_seed), np.random.default_rng(jitter_seed)
    rms = 0.0
    for i in range(a.n_frames):
        opd = make_ao_residual(pupil, a.residual_wfe_nm, phase_rng, a.seeing_arcsec,
                               a.reference_seeing_arcsec, a.outer_scale_m)
        tilt = tuple(jitter_rng.normal(0.0, a.tip_tilt_rms_mas, size=2))
        center = propagate(pupil, opd, a.wavelength_um, t.pad_factor, tilt)
        mean_psf += center.intensity / a.n_frames
        rms += rms_wfe_nm(opd, pupil) / a.n_frames
        if i == 0:
            first_frame, first_opd = center, opd
        for j, wavelength in enumerate(wavelengths):
            focal = propagate(pupil, opd, wavelength, t.pad_factor, tilt)
            smf[j] += overlap_efficiency(focal.field, gaussians[j]) / a.n_frames
            powers[j] += modal_coupling(focal.field, bases[j]) / a.n_frames
    # Overlaps are fractions of the sampled field's power; refer them to the
    # photons collected over the clear area (identical for a binary pupil).
    collected = pupil.sampled_power_fraction
    smf, powers = smf * collected, powers * collected
    return OpticalResult(
        wavelengths, smf, powers[:, :f.few_mode_count].sum(axis=1),
        f.lantern_transmission * powers[:, :f.lantern_ports].sum(axis=1), powers,
        mean_psf, reference, first_frame, first_opd, strehl_ratio(mean_psf, reference.intensity),
        marechal_strehl(rms, a.wavelength_um), rms, pupil.clear_area_m2,
    )


def run_simulation(config: SimulationConfig, *, noisy: bool = True) -> SimulationResult:
    """Run the configured optical ensemble and one photon/read-noise observation."""
    optics = simulate_optics(config)
    s, o, f = config.spectrograph, config.observation, config.fibre
    template, provenance = _template_with_provenance(config)
    model = SpectrographModel(template, s)
    efficiencies = {"smf": optics.smf_efficiency, "few_mode": optics.few_mode_efficiency,
                    "lantern": optics.lantern_efficiency}[f.feed]
    coupling = np.interp(model.wavelength_um, optics.wavelengths_um, efficiencies)
    detector_seed = np.random.SeedSequence(config.seed).spawn(3)[2]
    detector = simulate_detector(
        model(o.injected_rv_m_s), model.pixel_width_um, config.detector,
        collecting_area_m2=optics.collecting_area_m2, coupling=coupling,
        throughput=s.throughput, n_channels=f.lantern_ports if f.feed == "lantern" else 1,
        rng=np.random.default_rng(detector_seed), noisy=noisy,
    )
    rv = recover_rv(model, detector.flux_e, detector.variance_e2, continuum_e=detector.continuum_e,
                    search_min_m_s=-o.rv_search_half_width_m_s,
                    search_max_m_s=o.rv_search_half_width_m_s)
    return SimulationResult(config, optics, model, detector, rv, coupling, noisy, provenance)


def save_result(result: SimulationResult, directory: str | Path) -> Path:
    """Export reproducible configuration, provenance, spectra, optical arrays, and RV curve."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = result.summary()
    payload["configuration"] = asdict(result.config)
    payload["parameter_provenance"] = parameter_records(result.config)
    payload["software_versions"] = dict(
        python=platform.python_version(),
        **{name: version(name) for name in ("aophoton", "numpy", "scipy", "matplotlib", "PyYAML")})
    (directory / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    d = result.detector
    np.savetxt(directory / "spectrum.csv", np.column_stack([
        result.spectral_model.wavelength_um, d.expected_source_e, d.flux_e,
        d.variance_e2, d.snr, result.coupling]), delimiter=",",
        header=",".join(result.csv_columns()), comments="")
    np.savez_compressed(directory / "arrays.npz", psf=result.optics.mean_psf,
                        diffraction_psf=result.optics.reference.intensity,
                        axis_mas=result.optics.reference.axis_mas,
                        opd_m=result.optics.first_opd_m,
                        wavelengths_um=result.optics.wavelengths_um,
                        modal_power=result.optics.modal_power,
                        rv_grid_m_s=result.rv.grid_velocities_m_s, chi2=result.rv.chi2)
    return directory

