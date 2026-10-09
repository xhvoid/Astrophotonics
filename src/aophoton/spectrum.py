"""Dimensionless illustrative absorption templates, with wavelengths in micrometres."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

C_M_S = 299_792_458.0


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


def synthetic_spectrum(
    wavelength_min_um: float = 1.535,
    wavelength_max_um: float = 1.575,
    samples: int = 50_000,
    seed: int = 7,
    n_lines: int = 80,
) -> Spectrum:
    """Create seeded Gaussian absorption lines; no stellar type or line-list claim.

    Line depths (0.08--0.65) and intrinsic velocity sigmas (600--1600 m/s)
    are illustrative. The continuum is one, suitable for multiplying a photon
    spectral density [photons/s/m^2/um]. Overlapping transmission lines multiply.
    """
    if not np.isfinite([wavelength_min_um, wavelength_max_um]).all():
        raise ValueError("Wavelength bounds must be finite.")
    if not 0 < wavelength_min_um < wavelength_max_um:
        raise ValueError("Require 0 < wavelength_min_um < wavelength_max_um.")
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 3:
        raise ValueError("samples must be an integer >= 3.")
    if isinstance(n_lines, bool) or not isinstance(n_lines, (int, np.integer)) or n_lines < 0:
        raise ValueError("n_lines must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    log_wave = np.linspace(np.log(wavelength_min_um), np.log(wavelength_max_um), samples)
    flux = np.ones(samples)
    centers = rng.uniform(log_wave[0], log_wave[-1], n_lines)
    depths = rng.uniform(0.08, 0.65, n_lines)
    sigmas = rng.uniform(600.0, 1600.0, n_lines) / C_M_S
    for center, depth, sigma in zip(centers, depths, sigmas):
        flux *= 1.0 - depth * np.exp(-0.5 * ((log_wave - center) / sigma) ** 2)
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
    1D array with linear WAVE axis, CRVAL1/CDELT1/CRPIX1 and length CUNIT1.
    FITS flux units must be dimensionless or absent. Echelle products require a
    separate, instrument-specific extraction/conversion step.
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
        required = {"CRVAL1", "CDELT1", "CRPIX1", "CUNIT1", "CTYPE1"}
        if not required.issubset(header) or header["CTYPE1"].strip().upper() != "WAVE":
            raise ValueError("Primary FITS requires linear CTYPE1=WAVE and explicit CRVAL1/CDELT1/CRPIX1/CUNIT1.")
        flux_scale = dimensionless_scale(header.get("BUNIT"))
        indices = np.arange(hdu.data.size) + 1.0
        wavelength = header["CRVAL1"] + (indices - header["CRPIX1"]) * header["CDELT1"]
        wavelength = (wavelength * u.Unit(header["CUNIT1"])).to_value(u.um)
        return Spectrum(wavelength, flux_scale * hdu.data)
