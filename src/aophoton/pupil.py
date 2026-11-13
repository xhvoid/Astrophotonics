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
    ``amplitude`` is each pixel's open-area fraction: the pixel average of the
    aperture function, which is the field the FFT propagates. Photons are
    collected over ``clear_area_m2``; the sampled field's power is the smaller
    ``sampled_power_fraction`` of that wherever a pixel is partly covered.
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

    @property
    def clear_area_m2(self) -> float:
        """Geometric open area [m^2] that collects photons: sum(t) * pixel area."""
        return float(np.sum(self.amplitude) * self.pixel_scale_m**2)

    @property
    def sampled_power_fraction(self) -> float:
        """sum(t^2) / sum(t): power of the sampled pupil field per collected photon.

        Exactly 1 for a binary pupil. A partly covered pixel transmits t of
        the light falling on it, but its pixel-averaged field carries only
        t^2. Coupling efficiencies normalised to the sampled field are
        multiplied by this factor to refer them to the collected photons.
        """
        return float(np.sum(self.amplitude**2) / np.sum(self.amplitude))


def make_pupil(
    n_pix: int = 128,
    diameter_m: float = 3.6,
    obscuration: float = 0.25,
    spider_width_m: float = 0.0,
) -> Pupil:
    """Make an annular aperture with optional horizontal/vertical spiders.

    ``obscuration`` is the central-to-outer *diameter* ratio. Spiders are two
    orthogonal opaque bars of full width ``spider_width_m`` through the centre.
    Integer-centred samples align the optical axis with the FFT origin. Each
    pixel's spider transmission is its exact uncovered area fraction,
    (1 - c_x)(1 - c_y) for covered fractions c_x, c_y of its width and height,
    so vanes narrower than a pixel remove their true area instead of a whole
    row and column. The circular edges keep binary sampling, with the usual
    finite-grid discretization error.
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
    aperture = ((radius < diameter_m / 2) & (radius >= obscuration * diameter_m / 2)).astype(float)
    if spider_width_m > 0:
        open_fraction = 1.0 - _covered_fraction(axis, spider_width_m, pixel_scale)
        aperture *= open_fraction[None, :] * open_fraction[:, None]
    if np.count_nonzero(aperture) < 4:
        raise ValueError("aperture contains too few illuminated pixels")
    return Pupil(aperture, x_m, y_m, pixel_scale, float(diameter_m))


def _covered_fraction(centres: NDArray[np.float64], width: float, pixel: float) -> NDArray[np.float64]:
    """Fraction of each pixel [c - pixel/2, c + pixel/2] inside the bar |u| < width/2."""
    overlap = np.minimum(centres + pixel / 2, width / 2) - np.maximum(centres - pixel / 2, -width / 2)
    return np.clip(overlap / pixel, 0.0, 1.0)
