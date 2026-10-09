"""Poisson photon/electron statistics and extraction-aware Gaussian read noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DetectorConfig:
    """Exposure [s], continuum [photons/s/m^2/um], noise in electrons.

    Dark current and background are e-/s per detector extraction pixel per
    independently read spectral channel. QE is included in the spectrograph
    throughput, so each detected photon is counted as one electron. One exposure
    and one read per channel are assumed; no correlated or calibration noise.
    """

    exposure_s: float = 300.0
    continuum_photons_s_m2_um: float = 20_000_000.0
    read_noise_e: float = 5.0
    dark_current_e_s: float = 0.01
    background_e_s_pixel: float = 0.05
    extraction_pixels: int = 4

    def __post_init__(self) -> None:
        numbers = [self.exposure_s, self.continuum_photons_s_m2_um, self.read_noise_e,
                   self.dark_current_e_s, self.background_e_s_pixel]
        if not np.isfinite(numbers).all() or min(numbers) < 0 or self.exposure_s == 0:
            raise ValueError("Detector rates/noise must be finite and nonnegative; exposure must be positive.")
        if isinstance(self.extraction_pixels, bool) or not isinstance(self.extraction_pixels, (int, np.integer)) or self.extraction_pixels < 1:
            raise ValueError("extraction_pixels must be a positive integer.")


@dataclass(frozen=True)
class DetectorResult:
    """Background-subtracted electrons, known expectation/variance, and expected SNR."""

    flux_e: NDArray[np.float64]
    variance_e2: NDArray[np.float64]
    expected_source_e: NDArray[np.float64]
    continuum_e: NDArray[np.float64]
    snr: NDArray[np.float64]
    expected_background_e: float
    expected_dark_e: float
    read_variance_e2: float


def simulate_detector(
    normalized_flux: ArrayLike,
    pixel_width_um: ArrayLike,
    config: DetectorConfig,
    *,
    collecting_area_m2: float,
    coupling: ArrayLike,
    throughput: float,
    n_channels: int = 1,
    rng: np.random.Generator | None = None,
    noisy: bool = True,
) -> DetectorResult:
    """Sum co-aligned channels; retain their independent background/dark/read noise.

    Input relative flux is a *pixel-averaged* photon-density template. Total
    source counts use area [m^2], exposure [s], widths [um], throughput and total
    coupling (scalar or one value per spectral pixel). n_channels never
    multiplies the available source photons. Perfectly
    known mean dark/background are subtracted, but their shot variance remains.
    If noisy=False return the source expectation while retaining forecast variance.
    rng must be supplied when noisy=True, keeping reproducibility explicit.
    """
    flux = np.asarray(normalized_flux, dtype=float)
    widths = np.asarray(pixel_width_um, dtype=float)
    if flux.ndim != 1 or not flux.size or widths.shape != flux.shape:
        raise ValueError("Flux and widths must be matching nonempty one-dimensional arrays.")
    if not np.isfinite(flux).all() or not np.isfinite(widths).all() or np.any(flux < 0) or np.any(widths <= 0):
        raise ValueError("Flux must be nonnegative and widths positive, with finite values.")
    efficiencies = np.broadcast_to(np.asarray(coupling, dtype=float), flux.shape)
    if not np.isfinite([collecting_area_m2, throughput]).all() or not np.isfinite(efficiencies).all():
        raise ValueError("Area, coupling and throughput must be finite.")
    if collecting_area_m2 <= 0 or np.any((efficiencies < 0) | (efficiencies > 1)) or not 0 <= throughput <= 1:
        raise ValueError("Require positive area and coupling/throughput in [0,1].")
    if isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer)) or n_channels < 1:
        raise ValueError("n_channels must be a positive integer.")
    scale = config.continuum_photons_s_m2_um * collecting_area_m2 * config.exposure_s * throughput * efficiencies
    continuum = scale * widths
    source = continuum * flux
    extraction = config.extraction_pixels * n_channels
    dark = config.dark_current_e_s * config.exposure_s * extraction
    background = config.background_e_s_pixel * config.exposure_s * extraction
    read_variance = config.read_noise_e**2 * extraction
    variance = source + dark + background + read_variance
    if not np.isfinite(variance).all() or not np.isfinite(continuum).all():
        raise ValueError("Expected detector counts overflow; reduce flux or exposure.")
    if noisy:
        if rng is None:
            raise ValueError("Supply an explicit numpy Generator for a reproducible noisy observation.")
        measured = rng.poisson(source + dark + background).astype(float)
        measured += rng.normal(0.0, np.sqrt(read_variance), source.shape)
        measured -= dark + background
    else:
        measured = source.copy()
    snr = np.divide(source, np.sqrt(variance), out=np.zeros_like(source), where=variance > 0)
    return DetectorResult(measured, variance, source, continuum, snr, background, dark, read_variance)
