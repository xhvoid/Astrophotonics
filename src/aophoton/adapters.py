"""Explicit boundary for future OOPAO integration; no OOPAO validation claimed."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .pupil import Pupil


class OOPAOTelescopeLike(Protocol):
    """Minimal external telescope contract: ``OPD`` is a 2-D array in metres."""

    OPD: NDArray[np.float64]


@dataclass(frozen=True)
class OOPAOAdapter:
    """Import one externally computed OPD snapshot onto an identical pupil grid.

    This small adapter does not import, execute, or validate OOPAO. The
    caller must establish matching grid orientation, pupil transmission,
    pixel scale [m], and optical-path units [m]. ``telescope.OPD`` must be
    two-dimensional; a phase [rad] array must first be converted externally
    using its wavelength. No silent resizing or unit guessing is performed.

    Integration plan: run a documented OOPAO telescope/atmosphere/DM/WFS
    loop, export pupil and residual OPD snapshots, check WFE and energy,
    compare PSFs and Strehl on equal grids, then perform coupling statistics.
    That cross-package validation remains future work. OOPAO project:
    https://github.com/cheritier/OOPAO.
    """

    pupil: Pupil

    def residual_opd_m(self, telescope: OOPAOTelescopeLike) -> NDArray[np.float64]:
        """Copy and shape-check an externally generated OPD snapshot [m]."""
        opd = np.array(telescope.OPD, dtype=float, copy=True)
        if opd.shape != self.pupil.amplitude.shape or not np.all(np.isfinite(opd)):
            raise ValueError("external OPD must be finite and match the pupil grid")
        opd[~self.pupil.mask] = 0.0
        return opd
