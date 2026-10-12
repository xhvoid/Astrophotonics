"""Known-template weighted chi-square RV matching and local Fisher uncertainty."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

from .spectrum import C_M_S


@dataclass(frozen=True)
class RVResult:
    """Recovered RV/1-sigma [m/s], sampled chi-square, and search-boundary flag.

    ``reduced_chi2`` is ``minimum_chi2 / (N - 1)``: near one when the model and
    variance describe the data. A value far above one means the fit (and its
    Fisher uncertainty) should not be trusted, e.g. a wrong local minimum.
    """

    rv_m_s: float
    uncertainty_m_s: float
    grid_velocities_m_s: NDArray[np.float64]
    chi2: NDArray[np.float64]
    minimum_chi2: float
    at_boundary: bool
    reduced_chi2: float = float("nan")


# Central-difference step for the Fisher derivative; the model must cover the
# whole search interval widened by this step on each side.
FISHER_DERIVATIVE_STEP_M_S = 1.0
# Coarse-grid spacing as a fraction of the template's velocity correlation
# scale (see ``velocity_correlation_scale``); 0.5 keeps a grid node well inside
# the true chi-square basin for Gaussian-like features.
GRID_STEP_PER_CORRELATION_SCALE = 0.5
# Number of lowest coarse-grid local minima refined independently.
REFINED_CANDIDATES = 3
# Minimum number of velocities across the search at which the correlation scale
# is sampled; the smallest finite scale sets the grid spacing, because line
# content entering or leaving the band can make the template much sharper at
# some RVs than at the midpoint.  For a model exposing its pixel edges, probes
# are also spaced by at most PROBES_PER_BAND_WIDTH^-1 of the band's velocity
# width, so any template feature that enters the band during the search is
# inside it at some probe.
CORRELATION_SCALE_SAMPLES = 9
PROBES_PER_BAND_WIDTH = 2.0


def rv_uncertainty(
    model: Callable[[float], NDArray[np.float64]],
    rv_m_s: float,
    variance_e2: ArrayLike,
    continuum_e: ArrayLike = 1.0,
    derivative_step_m_s: float = FISHER_DERIVATIVE_STEP_M_S,
) -> float:
    """Known-continuum Fisher sigma = [sum((d counts/d RV)^2/variance)]^-1/2.

    A central difference at +/-1 m/s samples much less than a detector pixel.
    This is a local statistical forecast with fixed expected variance, neglecting
    template, wavelength-solution, continuum and instrumental calibration errors.
    """
    variance = np.asarray(variance_e2, dtype=float)
    continuum = np.broadcast_to(np.asarray(continuum_e, dtype=float), variance.shape)
    if not np.isfinite(derivative_step_m_s) or derivative_step_m_s <= 0:
        raise ValueError("The derivative step must be positive and finite [m/s].")
    if variance.ndim != 1 or not variance.size or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("RV variance must be a positive finite one-dimensional array.")
    if not np.isfinite(continuum).all() or np.any(continuum < 0):
        raise ValueError("Continuum electrons must be finite and nonnegative.")
    derivative = continuum * (model(rv_m_s + derivative_step_m_s) - model(rv_m_s - derivative_step_m_s)) / (2 * derivative_step_m_s)
    if derivative.shape != variance.shape or not np.isfinite(derivative).all():
        raise ValueError("Model must return finite flux matching the variance shape.")
    information = float(np.sum(derivative**2 / variance))
    return float(1.0 / np.sqrt(information)) if information > 0 else float("inf")


def velocity_correlation_scale(
    model: Callable[[float], NDArray[np.float64]],
    rv_m_s: float,
    variance_e2: ArrayLike,
    continuum_e: ArrayLike = 1.0,
    derivative_step_m_s: float = FISHER_DERIVATIVE_STEP_M_S,
) -> float:
    """Velocity scale [m/s] over which the template's chi-square basin decorrelates.

    ``sqrt(sum(w (m - <m>)^2) / sum(w (dm/dRV)^2))`` with ``w = continuum^2 /
    variance``: the weighted ratio of feature contrast to its velocity gradient.
    For a Gaussian feature of velocity width sigma it equals sqrt(2) sigma, and it
    is dominated by the sharpest weighted features. NaN if the model has no RV
    information.
    """
    variance = np.asarray(variance_e2, dtype=float)
    continuum = np.broadcast_to(np.asarray(continuum_e, dtype=float), variance.shape)
    weight = continuum**2 / variance
    centre = np.asarray(model(rv_m_s), dtype=float)
    derivative = (np.asarray(model(rv_m_s + derivative_step_m_s), dtype=float)
                  - np.asarray(model(rv_m_s - derivative_step_m_s), dtype=float)) / (2 * derivative_step_m_s)
    gradient = float(np.sum(weight * derivative**2))
    if not np.isfinite(gradient) or gradient <= 0:
        return float("nan")
    contrast = centre - np.sum(weight * centre) / np.sum(weight)
    return float(np.sqrt(np.sum(weight * contrast**2) / gradient))


def recover_rv(
    model: Callable[[float], NDArray[np.float64]],
    flux_e: ArrayLike,
    variance_e2: ArrayLike,
    *,
    continuum_e: ArrayLike = 1.0,
    search_min_m_s: float = -10_000.0,
    search_max_m_s: float = 10_000.0,
    grid_size: int = 81,
) -> RVResult:
    """Grid search then bounded local minimization; no injected RV is provided.

    model(RV) must apply the *same* LSF and pixel integration used in simulation.
    A known continuum electron baseline and fixed expected detector variance are
    assumed. This chi-square approximation is intended for adequate photon counts;
    photon-starved observations need a Poisson likelihood. Search must remain within
    model coverage, including 1 m/s on each side for the Fisher derivative.

    ``grid_size`` is a minimum: if its spacing exceeds
    ``GRID_STEP_PER_CORRELATION_SCALE`` times the template's velocity
    correlation scale (the smallest of the estimates at probes spread across the
    search -- at least ``CORRELATION_SCALE_SAMPLES``, and for a model with
    ``pixel_edges_um`` no further apart than half the band's velocity width),
    the coarse grid is densified so that a wide search does not step over the
    true chi-square basin. The ``REFINED_CANDIDATES`` lowest coarse local minima
    are each refined and the best is kept. This is a heuristic, not a
    global-optimum guarantee: always check ``reduced_chi2``.
    """
    flux = np.asarray(flux_e, dtype=float)
    variance = np.asarray(variance_e2, dtype=float)
    if flux.ndim != 1 or not flux.size or variance.shape != flux.shape:
        raise ValueError("RV flux and variance must be matching nonempty 1D arrays.")
    if not np.isfinite(flux).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("RV flux must be finite and variance strictly positive and finite.")
    continuum = np.broadcast_to(np.asarray(continuum_e, dtype=float), flux.shape)
    if not np.isfinite(continuum).all() or np.any(continuum < 0):
        raise ValueError("Continuum electrons must be finite and nonnegative.")
    if not np.isfinite([search_min_m_s, search_max_m_s]).all() or search_min_m_s >= search_max_m_s:
        raise ValueError("RV search bounds must be finite and increasing [m/s].")
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 5:
        raise ValueError("RV grid_size must be an integer >= 5.")

    def objective(rv: float) -> float:
        predicted = np.asarray(model(rv), dtype=float)
        if predicted.shape != flux.shape or not np.isfinite(predicted).all():
            raise ValueError("RV model must return finite flux with the observed shape.")
        return float(np.sum((flux - continuum * predicted)**2 / variance))

    # Stay one derivative step inside the bounds the model must support.
    n_probes = CORRELATION_SCALE_SAMPLES
    edges = getattr(model, "pixel_edges_um", None)
    if edges is not None and len(edges) >= 2 and edges[0] > 0 and edges[-1] > edges[0]:
        band_width_m_s = C_M_S * float(np.log(edges[-1] / edges[0]))
        n_probes = max(n_probes, int(np.ceil(PROBES_PER_BAND_WIDTH * (search_max_m_s - search_min_m_s)
                                             / band_width_m_s)) + 1)
    probes = np.linspace(search_min_m_s + FISHER_DERIVATIVE_STEP_M_S,
                         search_max_m_s - FISHER_DERIVATIVE_STEP_M_S, n_probes)
    scales = np.array([velocity_correlation_scale(model, float(v), variance, continuum) for v in probes])
    scales = scales[np.isfinite(scales) & (scales > 0)]
    scale = float(scales.min()) if scales.size else float("nan")
    if np.isfinite(scale) and scale > 0:
        needed = int(np.ceil((search_max_m_s - search_min_m_s) / (GRID_STEP_PER_CORRELATION_SCALE * scale))) + 1
        grid_size = max(int(grid_size), needed)
    velocities = np.linspace(search_min_m_s, search_max_m_s, grid_size)
    chi2 = np.array([objective(float(rv)) for rv in velocities])

    # Refine each of the lowest coarse local minima inside its bracketing cells.
    interior = (chi2[1:-1] <= chi2[:-2]) & (chi2[1:-1] <= chi2[2:])
    local = [0] * bool(chi2[0] <= chi2[1]) + list(np.flatnonzero(interior) + 1) \
        + [grid_size - 1] * bool(chi2[-1] <= chi2[-2])
    local = sorted(local, key=lambda i: chi2[i])[:REFINED_CANDIDATES]
    candidates = [(float(velocities[i]), float(chi2[i])) for i in local]
    for index in local:
        lower, upper = velocities[max(0, index - 1)], velocities[min(grid_size - 1, index + 1)]
        optimum = minimize_scalar(objective, bounds=(float(lower), float(upper)),
                                  method="bounded", options={"xatol": 1e-5})
        if not optimum.success:
            raise RuntimeError(f"RV minimization failed: {optimum.message}")
        candidates.append((float(optimum.x), float(optimum.fun)))
    rv, minimum = min(candidates, key=lambda item: item[1])
    sigma = rv_uncertainty(model, rv, variance, continuum)
    if not np.isfinite(sigma):
        raise ValueError("Spectrum carries no RV information: check source photons and template absorption structure.")
    # A best fit within one Fisher sigma of an edge is a truncated inference.
    boundary_margin = max(1.0, sigma)
    at_boundary = bool(min(rv - search_min_m_s, search_max_m_s - rv) <= boundary_margin)
    reduced_chi2 = minimum / max(flux.size - 1, 1)
    return RVResult(rv, sigma, velocities, chi2, minimum, at_boundary, reduced_chi2)
