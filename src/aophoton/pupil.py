"""Sampled circular telescope pupils; all pupil coordinates are in metres."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pupil:
    """Real amplitude transmission and Cartesian coordinates on a square grid.

    ``pixel_scale_m`` is the pupil sampling, not the focal-plane sampling.
    ``diameter_m`` describes the outer aperture before obscuration and spiders.
    """

    amplitude: NDArray[np.float64]
    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    pixel_scale_m: float
    diameter_m: float

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Pixels with nonzero amplitude transmission."""
        return self.amplitude > 0


def make_pupil(
    n_pix: int = 128,
    diameter_m: float = 3.6,
    obscuration: float = 0.25,
    spider_width_m: float = 0.0,
) -> Pupil:
    """Make an annular aperture with optional horizontal/vertical spiders.

    ``obscuration`` is the central-to-outer *diameter* ratio. Spiders are two
    orthogonal opaque bars of full width ``spider_width_m``. Integer-centred
    samples align the optical axis with the FFT origin. Binary edge sampling
    introduces the usual finite-grid pupil discretization error.
    """
    if not isinstance(n_pix, Integral) or isinstance(n_pix, bool) or n_pix < 8:
        raise ValueError("n_pix must be an integer >= 8")
    if not np.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("diameter_m must be finite and positive")
    if not np.isfinite(obscuration) or not 0 <= obscuration < 1:
        raise ValueError("obscuration must lie in [0, 1)")
    if not np.isfinite(spider_width_m) or not 0 <= spider_width_m < diameter_m:
        raise ValueError("spider_width_m must lie in [0, diameter_m)")
    pixel_scale = float(diameter_m / n_pix)
    axis = (np.arange(n_pix, dtype=float) - n_pix // 2) * pixel_scale
    x_m, y_m = np.meshgrid(axis, axis)
    radius = np.hypot(x_m, y_m)
    aperture = (radius < diameter_m / 2) & (radius >= obscuration * diameter_m / 2)
    if spider_width_m > 0:
        aperture &= (np.abs(x_m) >= spider_width_m / 2) & (
            np.abs(y_m) >= spider_width_m / 2
        )
    if np.count_nonzero(aperture) < 4:
        raise ValueError("aperture contains too few illuminated pixels")
    return Pupil(aperture.astype(float), x_m, y_m, pixel_scale, float(diameter_m))
