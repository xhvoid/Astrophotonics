"""Complex field overlap for a Gaussian single-mode fibre approximation.

See Shaklan & Roddier (1988), Applied Optics 27, 2334,
https://doi.org/10.1364/AO.27.002334. No intensity-only overlap is used.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .propagation import FocalPlane

# Sampling needed for a converged modal overlap. An unresolved mode collapses to
# a discrete delta whose overlap scales with the pixel size (1 mas at the
# default grid coupled 79x the continuous value), and a higher-order
# Hermite-Gaussian oscillates out to its turning point sqrt(2n + 1) (in units of
# the 1/e field radius / sqrt(2)), so the requirement grows with the degree n.
# Calibrated against padding-16 references for fields displaced by up to five
# mode radii and aberrated by up to 400 nm, at padding 2-4: the worst modal
# acceptance error falls below 0.1% at ~1.0 + 0.31 sqrt(2n + 1) focal-plane
# pixels per 1/e radius (1.3 px for n = 0, 2.2 px for n = 8); the base adds a
# 0.1 px margin (errors fall ~10x per 0.15 px).
SAMPLING_BASE_PIXELS = 1.1
SAMPLING_PIXELS_PER_TURNING_POINT = 0.31
# The grid half-width must also contain several radii so the mode is not truncated.
MIN_GRID_EXTENT_MODE_RADII = 3.0


def required_pixels_per_radius(max_degree: int = 0) -> float:
    """Focal-plane pixels per 1/e mode radius needed up to Hermite-Gaussian ``max_degree``."""
    return SAMPLING_BASE_PIXELS + SAMPLING_PIXELS_PER_TURNING_POINT * math.sqrt(2 * max_degree + 1)


def check_mode_sampling(focal: FocalPlane, mode_radius_mas: float,
                        extent_radii: float = MIN_GRID_EXTENT_MODE_RADII, max_degree: int = 0) -> None:
    """Raise ``ValueError`` unless modes up to ``max_degree`` are resolved and contained by ``focal``'s grid.

    The focal pixel scale is ``lambda / (pad_factor * D)``: raise
    ``telescope.pad_factor`` for a narrower or higher-order mode, or
    ``telescope.n_pix`` (grid extent ``~ n_pix/2 * lambda/D``) for a wider one.
    """
    pixels = mode_radius_mas / focal.pixel_scale_mas
    required = required_pixels_per_radius(max_degree)
    if pixels < required:
        raise ValueError(
            f"Mode radius {mode_radius_mas:g} mas spans {pixels:.2f} focal-plane pixels "
            f"({focal.pixel_scale_mas:.3g} mas); Hermite-Gaussian modes up to degree {max_degree} need "
            f"at least {required:.2f} for a converged (<0.1%) overlap -- increase telescope.pad_factor.")
    half_extent = float(min(-focal.axis_mas[0], focal.axis_mas[-1]))
    if half_extent < extent_radii * mode_radius_mas:
        raise ValueError(
            f"Focal grid half-width {half_extent:.4g} mas holds fewer than {extent_radii:g} "
            f"mode radii ({mode_radius_mas:g} mas); the mode is truncated -- increase telescope.n_pix.")


def gaussian_mode(focal: FocalPlane, mode_radius_mas: float) -> NDArray[np.float64]:
    """Unit discrete-energy Gaussian field on ``focal``'s angular grid.

    ``mode_radius_mas`` is the 1/e *field* radius, equal to the 1/e² intensity
    radius. Mode-field diameter is twice this value. An angular radius
    defines an equivalent injection mode; conversion to a physical fibre
    MFD needs an injection plate scale, which the MVP does not assume.
    """
    if not np.isfinite(mode_radius_mas) or mode_radius_mas <= 0:
        raise ValueError("mode_radius_mas must be finite and positive")
    check_mode_sampling(focal, mode_radius_mas)
    x, y = np.meshgrid(focal.axis_mas, focal.axis_mas)
    mode = np.exp(-(x**2 + y**2) / mode_radius_mas**2)
    norm = np.linalg.norm(mode)
    if norm <= np.finfo(float).tiny:
        raise ValueError("Gaussian mode is not sampled on this grid")
    return mode / norm


def overlap_efficiency(field: ArrayLike, mode: ArrayLike) -> float:
    """Normalized |sum(field * mode.conj())|² on the same uniform grid.

    Uniform pixel area cancels between numerator and denominator. Inputs
    may have arbitrary nonzero energy; field phase is essential. Returned
    values are clipped only for floating-point roundoff at the bounds.
    """
    signal = np.asarray(field, dtype=complex)
    accepted = np.asarray(mode, dtype=complex)
    if signal.shape != accepted.shape or signal.ndim != 2:
        raise ValueError("field and mode must have identical two-dimensional shapes")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(accepted)):
        raise ValueError("field and mode must be finite")
    signal_power = float(np.vdot(signal, signal).real)
    mode_power = float(np.vdot(accepted, accepted).real)
    if signal_power <= 0 or mode_power <= 0:
        raise ValueError("field and mode must have positive energy")
    efficiency = abs(np.vdot(accepted, signal))**2 / (signal_power * mode_power)
    return float(np.clip(efficiency, 0, 1))
