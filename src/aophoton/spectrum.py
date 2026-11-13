"""Dimensionless illustrative absorption templates, with wavelengths in micrometres."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

C_M_S = 299_792_458.0
# Intrinsic velocity sigmas [m/s] of the built-in template's lines.
LINE_SIGMA_RANGE_M_S = (600.0, 1600.0)
# Built-in template sampling: log-uniform, at least SAMPLES_PER_LINE_SIGMA
# samples per sigma of the narrowest possible line (200 m/s spacing), and no
# fewer than DEFAULT_TEMPLATE_SAMPLES.  Against templates refined to 1.6M
# samples, 3 per sigma fits a noiseless 2300 m/s injection to 2e-5 m/s at
# 1-2 um and R = 30,000 (Fisher sigma within 0.015%), where a fixed 50,000
# samples (4.2 km/s spacing) gave +29.9 m/s and +5.0% equivalent width.
SAMPLES_PER_LINE_SIGMA = 3.0
DEFAULT_TEMPLATE_SAMPLES = 50_000
# A line's transmission factor 1 - depth exp(-x^2/2) rounds to exactly 1.0
# beyond 8.6 sigma (depth <= 0.65), so evaluating it within this window only
# reproduces the full product bit for bit.
LINE_WINDOW_SIGMA = 10.0
# Header keywords that would make the primary (unlettered) 1D axis non-linear
# or couple it to other axes: projection parameters, distortion lookups,
# cross/extra-axis terms.  Matched whole: an alternate description's keywords
# end in its version letter A-Z (PC1_1A, PV1_1B, ...) and describe another WCS.
_UNSUPPORTED_WCS = re.compile(r"PV1_\d+|PS1_\d+|CPDIS1|CQDIS1|DP1(\..*)?|DQ1(\..*)?|D2IM\w*|(PC|CD)\d+_\d+")


@dataclass(frozen=True)
class Spectrum:
    """Rest-frame wavelength [um] and nonnegative, dimensionless relative flux."""

    wavelength_um: NDArray[np.float64]
    relative_flux: NDArray[np.float64]

    def __post_init__(self) -> None:
        wavelength = np.array(self.wavelength_um, dtype=float, copy=True)
        flux = np.array(self.relative_flux, dtype=float, copy=True)
        if wavelength.ndim != 1 or flux.shape != wavelength.shape or wavelength.size < 3:
            raise ValueError("Spectrum requires matching one-dimensional arrays of length >= 3.")
        if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(flux)):
            raise ValueError("Spectrum values must be finite.")
        if np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
            raise ValueError("Wavelengths [um] must be positive and strictly increasing.")
        if np.any(flux < 0):
            raise ValueError("Relative flux must be nonnegative.")
        wavelength.setflags(write=False)
        flux.setflags(write=False)
        object.__setattr__(self, "wavelength_um", wavelength)
        object.__setattr__(self, "relative_flux", flux)


def template_samples(wavelength_min_um: float, wavelength_max_um: float) -> int:
    """Samples of the built-in template over a wavelength range [um]: spacing
    of at most 1/SAMPLES_PER_LINE_SIGMA of the narrowest line sigma in
    log(lambda), and no fewer than DEFAULT_TEMPLATE_SAMPLES."""
    spacing = LINE_SIGMA_RANGE_M_S[0] / C_M_S / SAMPLES_PER_LINE_SIGMA
    needed = int(np.ceil(np.log(wavelength_max_um / wavelength_min_um) / spacing)) + 1
    return max(DEFAULT_TEMPLATE_SAMPLES, needed)


def synthetic_spectrum(
    wavelength_min_um: float = 1.535,
    wavelength_max_um: float = 1.575,
    samples: int | None = None,
    seed: int = 7,
    n_lines: int = 80,
) -> Spectrum:
    """Create seeded Gaussian absorption lines; no stellar type or line-list claim.

    Line depths (0.08--0.65) and intrinsic velocity sigmas (600--1600 m/s)
    are illustrative. The continuum is one, suitable for multiplying a photon
    spectral density [photons/s/m^2/um]. Overlapping transmission lines multiply.

    ``samples`` defaults to ``template_samples``, which resolves the narrowest
    line on any range (50,000 on the default one). An explicit count is used
    as given; below ``template_samples`` the template undersamples narrow lines.
    Line centres, depths and widths depend only on the seed, the range and
    ``n_lines``, never on the sample count.
    """
    if not np.isfinite([wavelength_min_um, wavelength_max_um]).all():
        raise ValueError("Wavelength bounds must be finite.")
    if not 0 < wavelength_min_um < wavelength_max_um:
        raise ValueError("Require 0 < wavelength_min_um < wavelength_max_um.")
    if samples is None:
        samples = template_samples(wavelength_min_um, wavelength_max_um)
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 3:
        raise ValueError("samples must be an integer >= 3.")
    if isinstance(n_lines, bool) or not isinstance(n_lines, (int, np.integer)) or n_lines < 0:
        raise ValueError("n_lines must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    log_wave = np.linspace(np.log(wavelength_min_um), np.log(wavelength_max_um), samples)
    flux = np.ones(samples)
    centers = rng.uniform(log_wave[0], log_wave[-1], n_lines)
    depths = rng.uniform(0.08, 0.65, n_lines)
    sigmas = rng.uniform(*LINE_SIGMA_RANGE_M_S, n_lines) / C_M_S
    for center, depth, sigma in zip(centers, depths, sigmas):
        low, high = np.searchsorted(log_wave, [center - LINE_WINDOW_SIGMA * sigma, center + LINE_WINDOW_SIGMA * sigma])
        window = log_wave[low:high]
        flux[low:high] *= 1.0 - depth * np.exp(-0.5 * ((window - center) / sigma) ** 2)
    return Spectrum(np.exp(log_wave), flux)


def doppler_factor(rv_m_s: float) -> float:
    """Relativistic longitudinal wavelength factor; positive RV is redshift."""
    if not np.isfinite(rv_m_s) or abs(rv_m_s) >= C_M_S:
        raise ValueError("RV must be finite with absolute value less than c [m/s].")
    beta = rv_m_s / C_M_S
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))


def load_spectrum(path: str | Path) -> Spectrum:
    """Read an explicit relative-flux schema; never infer wavelength units.

    ASCII: exactly two columns, wavelength [um] then relative flux (CSV or
    whitespace). FITS (optional Astropy): a table with WAVELENGTH and FLUX columns,
    with wavelength TUNIT convertible to um and dimensionless flux; or a primary
    1D array with a linear WAVE axis (see ``_linear_wave_axis``) and length
    CUNIT1. FITS flux units must be dimensionless or absent. Echelle products
    require a separate, instrument-specific extraction/conversion step.
    """
    path = Path(path)
    if path.suffix.lower() not in {".fits", ".fit", ".fts"}:
        data = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("ASCII template must contain exactly two columns: um, relative flux.")
        return Spectrum(data[:, 0], data[:, 1])

    try:
        from astropy import units as u
        from astropy.io import fits
    except ImportError as error:
        raise ImportError("FITS template import requires astropy.") from error

    def dimensionless_scale(unit: str | None) -> float:
        if unit and not u.Unit(unit).is_equivalent(u.dimensionless_unscaled):
            raise ValueError("FITS template flux must have dimensionless units.")
        return float(u.Unit(unit).to(u.dimensionless_unscaled)) if unit else 1.0

    with fits.open(path) as hdus:
        for hdu in hdus:
            names = getattr(getattr(hdu, "columns", None), "names", None)
            if names and {"WAVELENGTH", "FLUX"}.issubset(names):
                unit = hdu.columns["WAVELENGTH"].unit
                if not unit:
                    raise ValueError("FITS WAVELENGTH column requires an explicit length unit.")
                flux_scale = dimensionless_scale(hdu.columns["FLUX"].unit)
                wavelength = (np.asarray(hdu.data["WAVELENGTH"]) * u.Unit(unit)).to_value(u.um)
                return Spectrum(wavelength, flux_scale * np.asarray(hdu.data["FLUX"]))
        hdu = hdus[0]
        header = hdu.header
        if hdu.data is None or hdu.data.ndim != 1:
            raise ValueError("FITS requires a WAVELENGTH/FLUX table or a primary 1D WAVE array.")
        flux_scale = dimensionless_scale(header.get("BUNIT"))
        wavelength = _linear_wave_axis(header, hdu.data.size)
        unit = u.Unit(header["CUNIT1"])
        if not unit.is_equivalent(u.um):
            raise ValueError(f"Primary FITS CUNIT1={header['CUNIT1']!r} is not a length unit.")
        return Spectrum((wavelength * unit).to_value(u.um), flux_scale * hdu.data)


def _linear_wave_axis(header, size: int) -> NDArray[np.float64]:
    """Return the primary 1D WAVE axis in CUNIT1 units (FITS WCS Paper I, linear).

    world = CRVAL1 + s (p - CRPIX1) for 1-based pixel p, where s = CD1_1, or
    PC1_1 x CDELT1 (PC1_1 defaults to 1, and CDELT1 to 1 when PC1_1 is given).
    CD1_1 and PC1_1 together, WCSAXES other than 1, cross or extra-axis terms,
    projection parameters, distortion corrections and non-linear (WAVE-xxx) or
    non-wavelength CTYPE1 are refused, never ignored.  Only the primary
    description is read; alternate descriptions (keywords ending in A-Z) are
    neither used nor checked.
    """
    ctype = str(header.get("CTYPE1", "")).strip().upper()
    if ctype != "WAVE":
        found = f"CTYPE1={ctype!r}" if ctype else "no CTYPE1"
        raise ValueError(f"Primary FITS requires a linear vacuum-wavelength axis, CTYPE1='WAVE'; found {found}. "
                         "Non-linear algorithms (WAVE-LOG, WAVE-TAB, ...) and air wavelengths (AWAV) are not converted.")
    missing = [key for key in ("CRVAL1", "CRPIX1", "CUNIT1") if key not in header]
    if missing:
        raise ValueError(f"Primary FITS linear WAVE axis requires explicit {'/'.join(missing)}.")
    if header.get("WCSAXES", 1) != 1:
        raise ValueError(f"Primary FITS WAVE axis must be the only WCS axis; found WCSAXES={header['WCSAXES']}.")
    unsupported = sorted(key for key in header if _UNSUPPORTED_WCS.fullmatch(key) and key not in ("PC1_1", "CD1_1"))
    if unsupported:
        raise ValueError(f"Primary FITS WCS keywords {', '.join(unsupported)} are not supported for a linear 1D WAVE axis.")
    if "CD1_1" in header and "PC1_1" in header:
        raise ValueError("Primary FITS gives both CD1_1 and PC1_1; the FITS standard leaves this ambiguous.")
    if "CD1_1" in header:
        scale = float(header["CD1_1"])
    elif "CDELT1" in header or "PC1_1" in header:
        scale = float(header.get("PC1_1", 1.0)) * float(header.get("CDELT1", 1.0))
    else:
        raise ValueError("Primary FITS linear WAVE axis requires CD1_1, CDELT1 or PC1_1.")
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Primary FITS WAVE axis needs a finite, nonzero pixel scale.")
    pixel = np.arange(size) + 1.0
    return float(header["CRVAL1"]) + scale * (pixel - float(header["CRPIX1"]))
