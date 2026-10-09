"""Scalar Fraunhofer fields and incoherent, flux-conservative broadband PSFs."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from .pupil import Pupil

RAD_TO_MAS = 180 * 3600 * 1000 / np.pi


@dataclass(frozen=True)
class FocalPlane:
    """Uniform angular grid in milliarcseconds and unit-energy complex field.

    Field samples include the square root of pixel solid angle in their
    normalization: sum(abs(field)**2) = sum(intensity) = 1. Intensity is
    photon probability per pixel, not surface brightness per mas².
    """

    field: NDArray[np.complex128]
    intensity: NDArray[np.float64]
    axis_mas: NDArray[np.float64]
    pixel_scale_mas: float
    wavelength_um: float


@dataclass(frozen=True)
class BroadbandPSF:
    """Incoherent photon probability per pixel on one angular grid [mas].

    ``intensity.sum()`` is the captured photon fraction; cropping is never
    renormalized away. An incoherent wavelength mixture has no unique field.
    """

    intensity: NDArray[np.float64]
    axis_mas: NDArray[np.float64]
    pixel_scale_mas: float
    wavelengths_um: NDArray[np.float64]
    weights: NDArray[np.float64]


def propagate(
    pupil: Pupil,
    opd_m: ArrayLike,
    wavelength_um: float,
    pad_factor: int = 4,
    tip_tilt_mas: tuple[float, float] = (0.0, 0.0),
) -> FocalPlane:
    """Propagate OPD [m] to a unit-energy Fraunhofer field at wavelength [µm].

    ``tip_tilt_mas`` is the instantaneous (x, y) source displacement in
    milliarcseconds, introduced by a positive pupil phase ramp. Jitter must
    be drawn externally and its *intensities or coupled powers* averaged.
    The FFT convention makes positive x tilt move the image toward +x.
    Zero-padding increases angular sampling without adding pupil information.
    """
    opd = np.asarray(opd_m, dtype=float)
    if opd.shape != pupil.amplitude.shape or not np.all(np.isfinite(opd)):
        raise ValueError("opd_m must be finite and match the pupil shape")
    if not np.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("wavelength_um must be finite and positive")
    if not isinstance(pad_factor, Integral) or isinstance(pad_factor, bool) or pad_factor < 1:
        raise ValueError("pad_factor must be a positive integer")
    tilt = np.asarray(tip_tilt_mas, dtype=float)
    if tilt.shape != (2,) or not np.all(np.isfinite(tilt)):
        raise ValueError("tip_tilt_mas must contain two finite displacements")
    lambda_m = wavelength_um * 1e-6
    angular_nyquist_mas = lambda_m / (2 * pupil.pixel_scale_m) * RAD_TO_MAS
    if np.any(np.abs(tilt) >= angular_nyquist_mas):
        raise ValueError("tip-tilt exceeds the sampled angular Nyquist limit")
    ramp_m = (tilt[0] * pupil.x_m + tilt[1] * pupil.y_m) / RAD_TO_MAS
    entrance = pupil.amplitude * np.exp(2j * np.pi * (opd + ramp_m) / lambda_m)
    entrance /= np.sqrt(np.sum(pupil.amplitude**2))
    n_pix = pupil.amplitude.shape[0]
    n_fft = n_pix * pad_factor
    padded = np.zeros((n_fft, n_fft), dtype=complex)
    start = n_fft // 2 - n_pix // 2
    padded[start:start + n_pix, start:start + n_pix] = entrance
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded), norm="ortho"))
    intensity = np.abs(field)**2
    pixel_scale_mas = lambda_m / (n_fft * pupil.pixel_scale_m) * RAD_TO_MAS
    axis_mas = (np.arange(n_fft) - n_fft // 2) * pixel_scale_mas
    return FocalPlane(field, intensity, axis_mas, float(pixel_scale_mas), float(wavelength_um))


def _pixel_overlap_matrix(source: NDArray[np.float64], target: NDArray[np.float64]) -> csr_matrix:
    """Fraction of each source pixel assigned to each target pixel in 1-D."""
    ds, dt = source[1] - source[0], target[1] - target[0]
    source_low, source_high = source - ds / 2, source + ds / 2
    rows, columns, values = [], [], []
    for row, centre in enumerate(target):
        low, high = centre - dt / 2, centre + dt / 2
        first = np.searchsorted(source_high, low, side="right")
        last = np.searchsorted(source_low, high, side="left")
        for column in range(first, last):
            overlap = min(high, source_high[column]) - max(low, source_low[column])
            if overlap > 0:
                rows.append(row)
                columns.append(column)
                values.append(overlap / ds)
    return csr_matrix((values, (rows, columns)), shape=(len(target), len(source)))


def polychromatic_psf(
    pupil: Pupil,
    opd_m: ArrayLike,
    wavelengths_um: ArrayLike,
    weights: ArrayLike | None = None,
    pad_factor: int = 4,
    tip_tilt_mas: tuple[float, float] = (0.0, 0.0),
    axis_mas: ArrayLike | None = None,
) -> BroadbandPSF:
    """Sum photon-weighted monochromatic PSFs on one uniform angular grid.

    ``weights`` are incident photon weights, normalized internally; equal
    weights are used by default. The shared OPD is achromatic in metres.
    The default grid is that of the longest wavelength, enclosing the
    shorter-wave fields. Optional ``axis_mas`` must be ascending and uniform.
    Separable pixel-overlap resampling conserves discrete flux using a
    piecewise-constant pixel model. A cropped output retains its lost flux,
    so its sum can be less than one. This finite grid method is intended
    for a small wavelength range with sufficiently padded PSFs.
    """
    wavelengths = np.asarray(wavelengths_um, dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0 or not np.all(np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_um must be a nonempty vector of positive wavelengths")
    photon_weights = np.ones_like(wavelengths) if weights is None else np.asarray(weights, dtype=float)
    if photon_weights.shape != wavelengths.shape or not np.all(np.isfinite(photon_weights)) or np.any(photon_weights < 0) or photon_weights.sum() <= 0:
        raise ValueError("weights must match wavelengths and have nonnegative finite values with positive sum")
    photon_weights = photon_weights / photon_weights.sum()
    if axis_mas is None:
        target = propagate(pupil, opd_m, float(wavelengths.max()), pad_factor, tip_tilt_mas).axis_mas
    else:
        target = np.asarray(axis_mas, dtype=float)
        if target.ndim != 1 or target.size < 2 or not np.all(np.isfinite(target)):
            raise ValueError("axis_mas must be a finite one-dimensional grid")
        spacing = np.diff(target)
        if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-10, atol=0):
            raise ValueError("axis_mas must be uniformly spaced and ascending")
    result = np.zeros((target.size, target.size))
    for wavelength, weight in zip(wavelengths, photon_weights):
        focal = propagate(pupil, opd_m, float(wavelength), pad_factor, tip_tilt_mas)
        transform = _pixel_overlap_matrix(focal.axis_mas, target)
        result += weight * (transform @ (transform @ focal.intensity).T).T
    return BroadbandPSF(result, target, float(target[1] - target[0]), wavelengths, photon_weights)
