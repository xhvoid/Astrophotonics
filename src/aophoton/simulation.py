"""Seeded end-to-end orchestration, with independent optical and detector streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
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
from .spectrograph import SpectrographModel
from .spectrum import load_spectrum, synthetic_spectrum


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

    def summary(self) -> dict[str, Any]:
        """JSON-compatible scalar results; single-realization error is not an ensemble bias."""
        return dict(
            disclaimer="This is an independent conceptual study and is not an official ESO instrument model.",
            seed=self.config.seed, feed=self.config.fibre.feed,
            residual_wfe_nm=self.optics.residual_wfe_nm,
            measured_strehl=self.optics.strehl,
            marechal_without_jitter=self.optics.marechal,
            mean_smf_coupling=float(np.mean(self.optics.smf_efficiency)),
            mean_few_mode_coupling=float(np.mean(self.optics.few_mode_efficiency)),
            mean_lantern_coupling=float(np.mean(self.optics.lantern_efficiency)),
            mean_selected_coupling=float(np.mean(self.coupling)),
            collecting_area_m2=self.optics.collecting_area_m2,
            median_snr_per_pixel=float(np.median(self.detector.snr)),
            injected_rv_m_s=self.config.observation.injected_rv_m_s,
            recovered_rv_m_s=self.rv.rv_m_s,
            recovery_bias_m_s=self.rv.rv_m_s-self.config.observation.injected_rv_m_s,
            estimated_rv_uncertainty_m_s=self.rv.uncertainty_m_s,
            rv_at_search_boundary=bool(self.rv.at_boundary),
            rv_reduced_chi2=float(self.rv.reduced_chi2),
            validation="Synthetic only; no OOPAO or observational validation",
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
    return OpticalResult(
        wavelengths, smf, powers[:, :f.few_mode_count].sum(axis=1),
        f.lantern_transmission * powers[:, :f.lantern_ports].sum(axis=1), powers,
        mean_psf, reference, first_frame, first_opd, strehl_ratio(mean_psf, reference.intensity),
        marechal_strehl(rms, a.wavelength_um), rms,
        float(np.sum(pupil.amplitude**2) * pupil.pixel_scale_m**2),
    )


def run_simulation(config: SimulationConfig, *, noisy: bool = True) -> SimulationResult:
    """Run the configured optical ensemble and one photon/read-noise observation."""
    optics = simulate_optics(config)
    s, o, f = config.spectrograph, config.observation, config.fibre
    template = (load_spectrum(o.template_path) if o.template_path else
                synthetic_spectrum(s.wavelength_min_um-0.005, s.wavelength_max_um+0.005,
                                   seed=o.template_seed))
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
    return SimulationResult(config, optics, model, detector, rv, coupling)


def save_result(result: SimulationResult, directory: str | Path) -> Path:
    """Export reproducible configuration, provenance, spectra, optical arrays, and RV curve."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = result.summary()
    payload["configuration"] = asdict(result.config)
    payload["parameter_provenance"] = parameter_records(result.config)
    payload["software_versions"] = {name: version(name) for name in ("aophoton", "numpy", "scipy", "matplotlib", "PyYAML")}
    (directory / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    d = result.detector
    np.savetxt(directory / "spectrum.csv", np.column_stack([
        result.spectral_model.wavelength_um, d.expected_source_e, d.flux_e,
        d.variance_e2, d.snr, result.coupling]), delimiter=",",
        header="wavelength_um,expected_source_e,background_subtracted_noisy_e,variance_e2,snr_per_pixel,coupling",
        comments="")
    np.savez_compressed(directory / "arrays.npz", psf=result.optics.mean_psf,
                        diffraction_psf=result.optics.reference.intensity,
                        axis_mas=result.optics.reference.axis_mas,
                        opd_m=result.optics.first_opd_m,
                        wavelengths_um=result.optics.wavelengths_um,
                        modal_power=result.optics.modal_power,
                        rv_grid_m_s=result.rv.grid_velocities_m_s, chi2=result.rv.chi2)
    return directory

