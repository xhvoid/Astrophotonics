"""Complex field overlap for a Gaussian single-mode fibre approximation.

See Shaklan & Roddier (1988), Applied Optics 27, 2334,
https://doi.org/10.1364/AO.27.002334. No intensity-only overlap is used.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .propagation import FocalPlane

# A Gaussian field radius below ~1.25 focal-plane pixels is not resolved: a mode
# narrower than a pixel collapses to a discrete delta and its overlap scales
# with the pixel size (1 mas at the default grid coupled 79x the continuous
# value). At >= 1.25 px the ideal coupling is converged to ~0.1% under padding
# refinement. The grid half-width must also contain several radii so the mode
# is not truncated.
MIN_PIXELS_PER_MODE_RADIUS = 1.25
MIN_GRID_EXTENT_MODE_RADII = 3.0


def check_mode_sampling(focal: FocalPlane, mode_radius_mas: float, extent_radii: float = MIN_GRID_EXTENT_MODE_RADII) -> None:
    """Raise ``ValueError`` unless the mode is resolved and contained by ``focal``'s grid.

    The focal pixel scale is ``lambda / (pad_factor * D)``: raise
    ``telescope.pad_factor`` for a narrower mode, or ``telescope.n_pix`` (grid
    extent ``~ n_pix/2 * lambda/D``) for a wider one.
    """
    pixels = mode_radius_mas / focal.pixel_scale_mas
    if pixels < MIN_PIXELS_PER_MODE_RADIUS:
        raise ValueError(
            f"Mode radius {mode_radius_mas:g} mas spans {pixels:.2f} focal-plane pixels "
            f"({focal.pixel_scale_mas:.3g} mas); at least {MIN_PIXELS_PER_MODE_RADIUS} are needed "
            "for a converged overlap -- increase telescope.pad_factor.")
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
