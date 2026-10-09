"""Finite-grid von Kármán screens and a scalar residual-RMS AO approximation.

The absolute phase PSD uses cycles/m: W_phi(f) = 0.023 r0**(-5/3)
* (f**2 + L0**(-2))**(-11/6), in rad² m². See Fétick et al. (2018),
MNRAS 481, 5210, https://doi.org/10.1093/mnras/sty2595.
This is not an AO control-loop or spatial transfer-function simulator.
"""

import numpy as np
from numpy.typing import NDArray

from .pupil import Pupil

ARCSEC_TO_RAD = np.pi / (180 * 3600)


def von_karman_opd(
    pupil: Pupil,
    seeing_arcsec: float,
    rng: np.random.Generator,
    wavelength_um: float = 0.5,
    outer_scale_m: float = 25.0,
) -> NDArray[np.float64]:
    """Draw a PSD-normalized OPD screen [m] across the full pupil grid.

    Seeing is the conventional Kolmogorov-equivalent FWHM at
    ``wavelength_um``: r0 = 0.98 lambda / seeing. Real white noise supplies
    Hermitian Fourier coefficients, and sqrt(PSD)/pixel_scale supplies the
    two-dimensional frequency-bin normalization with unitary FFTs.

    The screen is periodic, omits piston, and has no subharmonics or inner
    scale. Frequencies below 1/(grid width) are absent: the finite-screen
    variance is not the infinite-aperture atmospheric variance. This routine
    does not force each realization to a specified RMS.
    """
    for name, value in (("seeing_arcsec", seeing_arcsec),
                        ("wavelength_um", wavelength_um),
                        ("outer_scale_m", outer_scale_m)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    n_pix = pupil.amplitude.shape[0]
    frequency = np.fft.fftfreq(n_pix, d=pupil.pixel_scale_m)
    fx, fy = np.meshgrid(frequency, frequency)
    lambda_m = wavelength_um * 1e-6
    r0_m = 0.98 * lambda_m / (seeing_arcsec * ARCSEC_TO_RAD)
    psd = 0.023 * r0_m ** (-5 / 3) * (
        fx**2 + fy**2 + outer_scale_m**-2
    ) ** (-11 / 6)
    psd[0, 0] = 0.0
    noise_fourier = np.fft.fft2(rng.normal(size=(n_pix, n_pix)), norm="ortho")
    phase_rad = np.fft.ifft2(
        noise_fourier * np.sqrt(psd) / pupil.pixel_scale_m, norm="ortho"
    ).real
    return phase_rad * lambda_m / (2 * np.pi)


def remove_piston_tiptilt(opd_m: NDArray[np.floating], pupil: Pupil) -> NDArray[np.float64]:
    """Remove a least-squares plane in OPD [m] over illuminated pupil pixels.

    The fit is weighted by intensity transmission. Pixels outside the pupil
    are set to zero. Removed pointing is represented separately by the
    propagation function's ``tip_tilt_mas`` argument.
    """
    opd = np.asarray(opd_m, dtype=float)
    if opd.shape != pupil.amplitude.shape or not np.all(np.isfinite(opd)):
        raise ValueError("opd_m must be finite and match the pupil shape")
    mask = pupil.mask
    design = np.column_stack((np.ones(mask.sum()), pupil.x_m[mask], pupil.y_m[mask]))
    sqrt_weight = pupil.amplitude[mask]
    coefficients = np.linalg.lstsq(
        design * sqrt_weight[:, None], opd[mask] * sqrt_weight, rcond=None
    )[0]
    result = np.zeros_like(opd)
    result[mask] = opd[mask] - design @ coefficients
    return result


def make_ao_residual(
    pupil: Pupil,
    residual_wfe_nm: float,
    rng: np.random.Generator,
    seeing_arcsec: float = 0.8,
    reference_seeing_arcsec: float = 0.8,
    outer_scale_m: float = 25.0,
) -> NDArray[np.float64]:
    """Make a piston/tip-tilt-removed residual optical path map [m].

    ``residual_wfe_nm`` is the requested intensity-weighted RMS at the
    reference seeing, both seeing values defined at 0.5 µm. For a fixed AO
    attenuation, actual RMS = residual_wfe_nm * (seeing/reference)**(5/6),
    following atmospheric phase-variance scaling with r0**(-5/3).

    Each finite screen is rescaled to that RMS; seeing is therefore a
    controlled scalar sensitivity parameter, not an independent PSF
    convolution. All spatial orders receive the same attenuation. No WFS,
    deformable mirror, temporal servo, or calibrated AO residual PSD is
    represented. Independent realizations describe an ensemble, not time.
    """
    if not np.isfinite(residual_wfe_nm) or residual_wfe_nm < 0:
        raise ValueError("residual_wfe_nm must be finite and nonnegative")
    for name, value in (("seeing_arcsec", seeing_arcsec),
                        ("reference_seeing_arcsec", reference_seeing_arcsec),
                        ("outer_scale_m", outer_scale_m)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if residual_wfe_nm == 0:
        return np.zeros_like(pupil.amplitude)
    opd = remove_piston_tiptilt(von_karman_opd(
        pupil, seeing_arcsec, rng, outer_scale_m=outer_scale_m
    ), pupil)
    weight = pupil.amplitude**2
    current_rms = np.sqrt(np.sum(weight * opd**2) / np.sum(weight))
    if current_rms <= np.finfo(float).tiny:
        raise ValueError("phase screen has zero residual variance")
    target_rms = residual_wfe_nm * 1e-9 * (seeing_arcsec / reference_seeing_arcsec) ** (5 / 6)
    return opd * (target_rms / current_rms)
