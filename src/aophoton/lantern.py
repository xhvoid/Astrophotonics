"""Reduced-order guided-mode acceptance and ideal photonic-lantern capture.

Hermite-Gaussian functions approximate a parabolic-index guide's modes;
they are not a calibrated step-index fibre or manufactured lantern model.
An ideal adiabatic lantern transfers the accepted modal subspace to at
least as many single-mode ports, conserving total modal power. Here the
first N modes define the accepted subspace and a scalar transmission
accounts for assumed insertion loss. Port/mode counts are per polarization.

See Birks et al. (2015), Advances in Optics and Photonics 7, 107,
https://doi.org/10.1364/AOP.7.000107. No random transfer matrix is used.
"""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_hermite

from .coupling import MIN_GRID_EXTENT_MODE_RADII, check_mode_sampling
from .propagation import FocalPlane


def mode_basis(focal: FocalPlane, mode_radius_mas: float, n_modes: int) -> NDArray[np.float64]:
    """Build deterministic, discrete orthonormal Hermite-Gaussian modes.

    Returns shape (n_modes, ny, nx). The fundamental has the same 1/e field
    radius [mas] as ``gaussian_mode``. Modes are ordered by total polynomial
    degree: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ... . Successive
    orthogonalization corrects finite-grid quadrature/truncation errors.
    Keeping a fixed radius and increasing n_modes creates nested subspaces.
    A partial degenerate shell is axis-dependent; complete shells (1, 3,
    6, 10, 15, ...) are preferable for rotation-symmetric comparisons.
    """
    if not isinstance(n_modes, Integral) or isinstance(n_modes, bool) or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    if n_modes > focal.field.size:
        raise ValueError("n_modes cannot exceed the number of grid samples")
    if not np.isfinite(mode_radius_mas) or mode_radius_mas <= 0:
        raise ValueError("mode_radius_mas must be finite and positive")
    # Hermite-Gaussian order n extends to ~sqrt(n + 1) fundamental radii.
    max_degree = int(np.ceil((np.sqrt(8 * n_modes + 1) - 1) / 2)) - 1
    check_mode_sampling(focal, mode_radius_mas,
                        MIN_GRID_EXTENT_MODE_RADII * float(np.sqrt(max_degree + 1)), max_degree=max_degree)
    x, y = np.meshgrid(focal.axis_mas / mode_radius_mas, focal.axis_mas / mode_radius_mas)
    envelope = np.exp(-(x**2 + y**2))
    basis: list[NDArray[np.float64]] = []
    degree = 0
    while len(basis) < n_modes:
        for nx in range(degree, -1, -1):
            ny = degree - nx
            mode = eval_hermite(nx, np.sqrt(2) * x) * eval_hermite(ny, np.sqrt(2) * y) * envelope
            initial_norm = np.linalg.norm(mode)
            if not np.isfinite(initial_norm) or initial_norm == 0:
                raise ValueError("mode order/radius is not numerically resolved on this grid")
            mode = mode / initial_norm
            for _ in range(2):
                for previous in basis:
                    mode -= np.sum(previous * mode) * previous
            norm = np.linalg.norm(mode)
            if norm < 1e-10:
                raise ValueError("mode order/radius is not numerically resolved on this grid")
            basis.append(mode / norm)
            if len(basis) == n_modes:
                break
        degree += 1
    return np.stack(basis)


def _validated_basis(basis: ArrayLike) -> NDArray:
    """Validate discrete orthonormality without altering supplied mode samples."""
    accepted = np.asarray(basis)
    if accepted.ndim != 3 or accepted.shape[0] == 0:
        raise ValueError("basis must have shape (n_modes, ny, nx)")
    if not np.all(np.isfinite(accepted)):
        raise ValueError("basis must be finite")
    flattened = accepted.reshape(accepted.shape[0], -1)
    gram = flattened.conj() @ flattened.T
    if not np.allclose(gram, np.eye(len(accepted)), rtol=0, atol=1e-9):
        raise ValueError("basis must be orthonormal on the discrete grid")
    return accepted


def _project(
    field: ArrayLike,
    conjugate_modes: NDArray,
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Project onto already validated, conjugated flattened mode samples."""
    signal = np.asarray(field, dtype=complex)
    if signal.shape != shape or not np.all(np.isfinite(signal)):
        raise ValueError("field must be finite and match the basis grid")
    power = float(np.vdot(signal, signal).real)
    if power <= 0:
        raise ValueError("field must have positive energy")
    powers = np.abs(conjugate_modes @ signal.ravel())**2 / power
    if powers.sum() > 1 + 1e-9:
        raise ValueError("modal projection exceeds incident optical power")
    return np.clip(powers.real, 0, 1)


class ModeProjector:
    """Reusable projector with a privately copied, validated immutable basis.

    Use ``ModeProjector(basis).powers(field)`` for repeated realizations on
    one grid. Orthonormality is checked once when the object is constructed;
    field shape, finite values, and incident energy are checked every time.
    No global cache or random state is used. Changing the original basis
    after construction cannot change this projector's accepted modes.
    """

    def __init__(self, basis: ArrayLike) -> None:
        accepted = _validated_basis(np.array(basis, copy=True))
        self._shape = accepted.shape[1:]
        self._conjugate_modes = accepted.reshape(accepted.shape[0], -1).conj()
        self._conjugate_modes.setflags(write=False)

    def powers(self, field: ArrayLike) -> NDArray[np.float64]:
        """Return projected modal fractions of incident field power, unitless."""
        return _project(field, self._conjugate_modes, self._shape)


def modal_coupling(field: ArrayLike, basis: ArrayLike) -> NDArray[np.float64]:
    """Project a complex field onto an orthonormal basis; return power fractions.

    Input basis must have shape (n_modes, ny, nx). Powers are independent of
    incident field normalization. Invalid/nonorthonormal bases are rejected,
    because summing their overlaps would double-count optical power. For
    repeated fields, ``ModeProjector`` checks the fixed basis only once.
    """
    accepted = _validated_basis(basis)
    return _project(field, accepted.reshape(len(accepted), -1).conj(), accepted.shape[1:])


def lantern_efficiency(
    field: ArrayLike,
    basis: ArrayLike,
    ports: int,
    transmission: float = 0.9,
) -> float:
    """Total accepted power in first ``ports`` modes times assumed transmission.

    ``ports`` counts spatial modes for one polarization. ``transmission``
    is a scalar power transmission [0, 1], not a predicted taper loss. Using
    transmission=1 gives the corresponding ideal few-mode acceptance.
    """
    accepted = np.asarray(basis)
    if not isinstance(ports, Integral) or isinstance(ports, bool) or ports < 1 or accepted.ndim != 3 or ports > len(accepted):
        raise ValueError("ports must select at least one and no more than the available modes")
    if not np.isfinite(transmission) or not 0 <= transmission <= 1:
        raise ValueError("transmission must lie in [0, 1]")
    return float(np.clip(modal_coupling(field, accepted[:ports]).sum() * transmission, 0, 1))
