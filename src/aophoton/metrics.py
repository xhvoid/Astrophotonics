"""Wavefront and PSF diagnostics with explicit normalization."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pupil import Pupil


def rms_wfe_nm(opd_m: ArrayLike, pupil: Pupil) -> float:
    """Intensity-weighted, piston-removed pupil RMS optical path error [nm]."""
    opd = np.asarray(opd_m, dtype=float)
    if opd.shape != pupil.amplitude.shape or not np.all(np.isfinite(opd)):
        raise ValueError("opd_m must be finite and match the pupil shape")
    weight = pupil.amplitude**2
    mean = np.sum(weight * opd) / np.sum(weight)
    return float(1e9 * np.sqrt(np.sum(weight * (opd - mean)**2) / np.sum(weight)))


def marechal_strehl(wfe_nm: ArrayLike, wavelength_um: float) -> float | NDArray[np.float64]:
    """Return exp[-(2 pi sigma/lambda)²]; reliable for small phase variance.

    WFE is RMS optical path in nanometres and wavelength is in micrometres.
    This approximation is a cross-check, not a substitute for propagation.
    """
    wfe = np.asarray(wfe_nm, dtype=float)
    if not np.all(np.isfinite(wfe)) or np.any(wfe < 0):
        raise ValueError("wfe_nm must be finite and nonnegative")
    if not np.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("wavelength_um must be finite and positive")
    result = np.exp(-(2 * np.pi * wfe / (wavelength_um * 1000))**2)
    return float(result) if result.ndim == 0 else result


def strehl_ratio(intensity: ArrayLike, reference_intensity: ArrayLike) -> float:
    """Ratio of flux-normalized sampled peaks on identical angular grids.

    The reference must use the same pupil, wavelength, sampling, and field
    of view. Taking the maximum measures the image peak; a static pointing
    offset is not itself a wavefront-quality penalty. No clipping to unity
    is applied, so incorrect references and sampling errors remain visible.
    """
    psf = np.asarray(intensity, dtype=float)
    reference = np.asarray(reference_intensity, dtype=float)
    if psf.shape != reference.shape or psf.ndim != 2:
        raise ValueError("PSFs must have identical two-dimensional shapes")
    for array in (psf, reference):
        if not np.all(np.isfinite(array)) or np.any(array < 0) or array.sum() <= 0:
            raise ValueError("PSFs must be finite, nonnegative, and have positive flux")
    return float((psf.max() / psf.sum()) / (reference.max() / reference.sum()))
