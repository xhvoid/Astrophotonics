"""Constant-resolving-power Gaussian LSF and finite detector-pixel integration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from .spectrum import C_M_S, Spectrum, doppler_factor


@dataclass(frozen=True)
class SpectrographConfig:
    """Illustrative parameters; wavelengths [um], RV padding [m/s], throughput [0,1]."""

    wavelength_min_um: float = 1.54
    wavelength_max_um: float = 1.57
    resolving_power: float = 100_000.0
    pixels_per_resolution_element: float = 3.0
    oversample: int = 4
    throughput: float = 0.15
    rv_padding_m_s: float = 30_000.0

    def __post_init__(self) -> None:
        numbers = [self.wavelength_min_um, self.wavelength_max_um, self.resolving_power,
                   self.pixels_per_resolution_element, self.throughput, self.rv_padding_m_s]
        if not np.isfinite(numbers).all():
            raise ValueError("Spectrograph parameters must be finite.")
        if not 0 < self.wavelength_min_um < self.wavelength_max_um:
            raise ValueError("Require positive increasing wavelength bounds [um].")
        if self.resolving_power <= 0 or self.pixels_per_resolution_element < 2:
            raise ValueError("Resolving power must be positive and sampling >= 2 pixels/FWHM.")
        if isinstance(self.oversample, bool) or not isinstance(self.oversample, (int, np.integer)) or self.oversample < 2:
            raise ValueError("oversample must be an integer >= 2.")
        if not 0 <= self.throughput <= 1 or not 0 < self.rv_padding_m_s < C_M_S:
            raise ValueError("Require throughput in [0,1] and positive subluminal RV padding.")


@dataclass(frozen=True)
class SpectrographOutput:
    """Pixel means and high-resolution dimensionless flux, with wavelengths in um."""

    wavelength_um: NDArray[np.float64]
    pixel_width_um: NDArray[np.float64]
    normalized_flux: NDArray[np.float64]
    highres_wavelength_um: NDArray[np.float64]
    input_flux: NDArray[np.float64]
    convolved_flux: NDArray[np.float64]


class SpectrographModel:
    """Shift a rest template, convolve in log(lambda), then integrate each pixel.

    The pre-pixel LSF has FWHM(log lambda)=1/R. Finite pixel integration further
    broadens the measured profile. Six-sigma LSF padding and explicit template
    coverage at both RV extremes prevent extrapolation or edge-derived RV signal.
    The same callable is used to inject and recover RV. No Doppler beaming or
    absolute flux transformation is included: RV shifts the normalized template.
    """

    def __init__(self, template: Spectrum, config: SpectrographConfig = SpectrographConfig()):
        self.template = template
        self.config = config
        log_min, log_max = np.log([config.wavelength_min_um, config.wavelength_max_um])
        n_pixels = int(np.ceil((log_max - log_min) * config.resolving_power * config.pixels_per_resolution_element))
        self.pixel_edges_um = np.exp(np.linspace(log_min, log_max, n_pixels + 1))
        self.wavelength_um = np.sqrt(self.pixel_edges_um[:-1] * self.pixel_edges_um[1:])
        self.pixel_width_um = np.diff(self.pixel_edges_um)
        self.dlog_pixel = (log_max - log_min) / n_pixels
        self.lsf_sigma_log = 1.0 / (config.resolving_power * np.sqrt(8.0 * np.log(2.0)))
        self._dlog_high = self.dlog_pixel / config.oversample
        self._sigma_samples = self.lsf_sigma_log / self._dlog_high
        self._pad = int(np.ceil(6.0 * self._sigma_samples)) + 2
        self._n_intervals = n_pixels * config.oversample
        indices = np.arange(-self._pad, self._n_intervals + self._pad + 1)
        self._high_wavelength_um = np.exp(log_min + indices * self._dlog_high)
        low = self._high_wavelength_um[0] / doppler_factor(config.rv_padding_m_s)
        high = self._high_wavelength_um[-1] / doppler_factor(-config.rv_padding_m_s)
        if template.wavelength_um[0] > low or template.wavelength_um[-1] < high:
            raise ValueError(f"Template must cover at least [{low:.9f}, {high:.9f}] um including LSF and RV padding.")
        self._core = slice(self._pad, self._pad + self._n_intervals + 1)
        self.highres_wavelength_um = self._high_wavelength_um[self._core]
        self._interval_width_um = np.diff(self.highres_wavelength_um)

    def _forward(self, rv_m_s: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if not np.isfinite(rv_m_s) or abs(rv_m_s) > self.config.rv_padding_m_s:
            raise ValueError("RV [m/s] exceeds configured template padding.")
        rest_wavelength = self._high_wavelength_um / doppler_factor(rv_m_s)
        shifted = np.interp(rest_wavelength, self.template.wavelength_um, self.template.relative_flux)
        convolved = gaussian_filter1d(shifted, self._sigma_samples, mode="nearest", truncate=6.0)
        core = convolved[self._core]
        integral = 0.5 * (core[:-1] + core[1:]) * self._interval_width_um
        pixel_means = integral.reshape(-1, self.config.oversample).sum(axis=1) / self.pixel_width_um
        return pixel_means, shifted[self._core], core

    def __call__(self, rv_m_s: float) -> NDArray[np.float64]:
        """Return relative flux integrated over detector pixels at RV [m/s]."""
        return self._forward(rv_m_s)[0]

    def evaluate(self, rv_m_s: float) -> SpectrographOutput:
        """Return input, convolved and detector-sampled spectra at RV [m/s]."""
        sampled, shifted, convolved = self._forward(rv_m_s)
        return SpectrographOutput(self.wavelength_um, self.pixel_width_um, sampled,
                                 self.highres_wavelength_um, shifted, convolved)
