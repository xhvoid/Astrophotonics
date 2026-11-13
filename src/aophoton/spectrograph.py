"""Constant-resolving-power Gaussian LSF and finite detector-pixel integration."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from .spectrum import Spectrum, doppler_factor

# The Gaussian LSF is truncated, then renormalised, this many sigma from its centre.
LSF_TRUNCATE_SIGMA = 6.0
# Internal-grid nodes kept beyond every position an in-padding RV can reach, so
# spline evaluation never uses its end intervals.
SPLINE_MARGIN_NODES = 4
# Merged template/grid intervals projected per block (bounds temporary memory).
PROJECTION_BLOCK = 1 << 20
# Attenuation per node of a change in the data of a cubic interpolating spline
# on uniform nodes (the root of x^2 + 4x + 1 inside the unit circle).
SPLINE_COUPLING = 2.0 - np.sqrt(3.0)


@dataclass(frozen=True)
class SpectrographConfig:
    """Illustrative parameters; wavelengths [um], RV padding [m/s], throughput [0,1].

    ``oversample`` is the number of internal-grid steps per detector pixel. The
    template is projected onto that grid and the LSF is sampled on it; the
    result converges as (step / LSF sigma)^2, independently of how narrow the
    template's own lines are (see ``SpectrographModel``).
    """

    wavelength_min_um: float = 1.54
    wavelength_max_um: float = 1.57
    resolving_power: float = 100_000.0
    pixels_per_resolution_element: float = 3.0
    oversample: int = 4
    throughput: float = 0.15
    rv_padding_m_s: float = 30_000.0

    def __post_init__(self) -> None:
        numbers = [self.wavelength_min_um, self.wavelength_max_um, self.resolving_power,
                   self.pixels_per_resolution_element, self.throughput, self.rv_padding_m_s]
        if not np.isfinite(numbers).all():
            raise ValueError("Spectrograph parameters must be finite.")
        if not 0 < self.wavelength_min_um < self.wavelength_max_um:
            raise ValueError("Require positive increasing wavelength bounds [um].")
        if self.resolving_power <= 0 or self.pixels_per_resolution_element < 2:
            raise ValueError("Resolving power must be positive and sampling >= 2 pixels/FWHM.")
        if isinstance(self.oversample, bool) or not isinstance(self.oversample, (int, np.integer)) or self.oversample < 2:
            raise ValueError("oversample must be an integer >= 2.")
        if not 0 <= self.throughput <= 1 or not 0 < self.rv_padding_m_s < 299_792_458.0:
            raise ValueError("Require throughput in [0,1] and positive subluminal RV padding.")


@dataclass(frozen=True)
class SpectrographOutput:
    """Pixel means and internal-grid dimensionless flux, with wavelengths in um.

    ``highres_wavelength_um`` is the model's fixed observer-frame grid, the
    same at every RV. ``input_flux`` is the shifted template on it as the model
    uses it (its projection onto the internal grid); ``convolved_flux`` adds
    the LSF.
    """

    wavelength_um: NDArray[np.float64]
    pixel_width_um: NDArray[np.float64]
    normalized_flux: NDArray[np.float64]
    highres_wavelength_um: NDArray[np.float64]
    input_flux: NDArray[np.float64]
    convolved_flux: NDArray[np.float64]


@dataclass(frozen=True)
class _Layout:
    """Pixel grid and internal grid shared by every RV evaluation."""

    log_min: float
    n_pixels: int
    dlog_pixel: float
    step: float
    lsf_sigma_log: float
    kernel_sigma_log: float
    first: int          # internal-grid nodes are log_min + k * step, first <= k <= last
    last: int
    reach: int          # LSF half-width in nodes

    @property
    def support_log(self) -> tuple[float, float]:
        """Rest-frame log-wavelength interval the template must cover."""
        return (self.log_min + (self.first - self.reach - 1) * self.step,
                self.log_min + (self.last + self.reach + 1) * self.step)


def _layout(config: SpectrographConfig) -> _Layout:
    log_min, log_max = np.log([config.wavelength_min_um, config.wavelength_max_um])
    n_pixels = int(np.ceil((log_max - log_min) * config.resolving_power * config.pixels_per_resolution_element))
    dlog_pixel = (log_max - log_min) / n_pixels
    step = dlog_pixel / config.oversample
    sigma = 1.0 / (config.resolving_power * np.sqrt(8.0 * np.log(2.0)))
    # Projection onto hat functions adds a mean variance step^2/6 (see
    # SpectrographModel); the sampled kernel supplies the rest of sigma^2.
    kernel_sigma = float(np.sqrt(sigma**2 - step**2 / 6.0))
    shift = np.log(doppler_factor(config.rv_padding_m_s)) / step       # largest |log Doppler factor| in steps
    first = int(np.floor(-shift)) - SPLINE_MARGIN_NODES
    last = int(np.ceil(n_pixels * config.oversample + shift)) + SPLINE_MARGIN_NODES
    reach = int(np.ceil(LSF_TRUNCATE_SIGMA * kernel_sigma / step))
    return _Layout(float(log_min), n_pixels, float(dlog_pixel), float(step), float(sigma), kernel_sigma,
                   first, last, reach)


def template_support_um(config: SpectrographConfig) -> tuple[float, float]:
    """Rest-frame wavelength interval [um] a template must cover for ``config``.

    It spans the band, the Doppler shift of every RV within
    ``rv_padding_m_s``, the truncated LSF and the projection grid.  The bounds
    are rounded outward to whole floating-point steps, so that their logarithms
    bracket the model's log-wavelength support; a template whose first and last
    samples are exactly these values is accepted by ``SpectrographModel``.
    """
    low, high = _layout(config).support_log
    low_um, high_um = float(np.exp(low)), float(np.exp(high))
    while np.log(low_um) > low:
        low_um = float(np.nextafter(low_um, 0.0))
    while np.log(high_um) < high:
        high_um = float(np.nextafter(high_um, np.inf))
    return low_um, high_um


def _hat_projection(x: NDArray[np.float64], values: NDArray[np.float64], count: int) -> NDArray[np.float64]:
    """Exact integrals of the piecewise-linear interpolant of (x, values)
    against unit hat functions centred on the integer nodes 0..count-1.

    ``x`` is in units of the node spacing and must cover [-1, count]. Every
    merged interval of knots and nodes carries the product of two linear
    functions, which Simpson's rule integrates exactly; nothing is point
    sampled, so area and centroid are preserved for features of any width.
    """
    inner = x[(x > -1) & (x < count)]
    points = np.sort(np.concatenate([np.arange(-1.0, count + 1.0), inner]))
    flux = np.interp(points, x, values)
    total = np.zeros(count + 1)
    for start in range(0, points.size - 1, PROJECTION_BLOCK):
        stop = min(start + PROJECTION_BLOCK, points.size - 1)
        a, b, fa, fb = points[start:stop], points[start + 1:stop + 1], flux[start:stop], flux[start + 1:stop + 1]
        cell = np.clip(np.floor(0.5 * (a + b)).astype(int) + 1, 0, count)       # interval [cell - 1, cell]
        ta, tb = a - (cell - 1), b - (cell - 1)
        rising = (b - a) / 6.0 * (fa * ta + (fa + fb) * (ta + tb) + fb * tb)
        falling = 0.5 * (b - a) * (fa + fb) - rising
        total += np.bincount(cell, rising, count + 1) + np.append(np.bincount(cell, falling, count + 1)[1:], 0.0)
    return total[:count]


class SpectrographModel:
    """Shift a rest template, convolve in log(lambda), then integrate each pixel.

    The pre-pixel LSF has FWHM(log lambda)=1/R. Finite pixel integration further
    broadens the measured profile. Six-sigma LSF padding and explicit template
    coverage at both RV extremes prevent extrapolation or edge-derived RV signal.
    The same callable is used to inject and recover RV. No Doppler beaming or
    absolute flux transformation is included: RV shifts the normalized template.

    A constant-R LSF and log-uniform pixels make the chain translation invariant
    in log(lambda), so the template is convolved once, at construction, in its
    rest frame: its piecewise-linear interpolant is projected exactly onto hat
    functions of the internal grid (no point sampling, so lines narrower than a
    grid step keep their area and centroid) and correlated with the sampled
    Gaussian. Each RV then integrates a cubic spline of the convolved depth,
    weighted by d(lambda), exactly over the Doppler-shifted pixel edges, which
    conserves the line flux for every shift. Two errors remain. A line
    narrower than the grid step gets a projection variance set by where it
    falls between rest-frame nodes, so its LSF variance is off by at most
    step^2/6, whatever the RV. Spline interpolation adds an error of order
    (step/sigma)^4. Construction cost is linear in the template samples it
    covers; each RV evaluation costs O(pixels) whatever the template density.

    Only the template samples inside the support (``template_support_um``) and
    the two that bracket it are read: they fix the template's interpolant
    wherever the LSF, a pixel or an allowed RV can reach, and no other sample
    can change the model.  Depth is taken from the reference level nearest one
    within that range.  Each pixel integral is summed from its own spline
    intervals, so roundoff stays relative to that pixel's absorption rather
    than to the cumulative integral of the whole band, and it is divided by the
    integral of the same weight, d(lambda), over exactly the span those
    intervals cover (``exp(start) * expm1(end - start)``): the mean of a
    constant depth is then that depth to within a few roundings at every RV.
    (Differencing the pixel edges' exponentials instead loses about 6e-11 of
    the width to cancellation, and the shifted bounds round differently from
    them.)

    A pixel's value depends only on the template near its rest-frame interval:
    within the truncated LSF and the projection's one-node hat, plus the nodes
    over which the global spline still couples a change at double precision
    (it is attenuated by ``SPLINE_COUPLING`` per node; ``_flat_margin``).  Where
    every template sample in that window has one value, the exact model returns
    that value, so the pixel takes it exactly: a locally flat stretch of the
    template yields a flux that is constant to the last bit at every RV that
    keeps it in view, and so no RV information, however different the template
    is elsewhere in the padding.  Any differing sample in the window, however
    shallow, leaves the pixel to the integration above.

    ``highres_wavelength_um`` is the fixed observer-frame internal grid, from
    the first to the last pixel edge in steps of ``dlog_pixel / oversample``;
    ``evaluate`` returns its diagnostic spectra on it at every RV.
    """

    def __init__(self, template: Spectrum, config: SpectrographConfig = SpectrographConfig()):
        self.template = template
        self.config = config
        layout = self._layout = _layout(config)
        log_min = layout.log_min
        relative_edges = np.linspace(0.0, layout.n_pixels * layout.dlog_pixel, layout.n_pixels + 1)
        self.pixel_edges_um = np.exp(log_min + relative_edges)
        self.wavelength_um = np.sqrt(self.pixel_edges_um[:-1] * self.pixel_edges_um[1:])
        self.pixel_width_um = np.diff(self.pixel_edges_um)
        self.dlog_pixel = layout.dlog_pixel
        self.lsf_sigma_log = layout.lsf_sigma_log
        self._grid = np.arange(layout.n_pixels * config.oversample + 1) * layout.step    # log(lambda / lambda_min)
        self.highres_wavelength_um = np.exp(log_min + self._grid)
        low_um, high_um = template_support_um(config)
        wave = template.wavelength_um
        if wave[0] > low_um or wave[-1] < high_um:
            raise ValueError(f"Template must cover at least [{low_um:.9f}, {high_um:.9f}] um including LSF and RV padding.")
        # The samples inside the support and the two that bracket it.
        reachable = slice(int(np.searchsorted(wave, low_um, side="right")) - 1,
                          int(np.searchsorted(wave, high_um, side="left")) + 1)
        knots, flux = np.log(wave[reachable]), template.relative_flux[reachable]
        # Nodes first-reach .. last+reach, in units of the grid step relative to node first-reach.
        origin = layout.first - layout.reach
        count = layout.last - layout.first + 1 + 2 * layout.reach
        self._flux_bounds = (float(flux.min()), float(flux.max()))
        # A constant offset integrates exactly. Keep unity for normalized spectra
        # and the nearest reachable level for templates entirely above or below it.
        self._continuum = float(np.clip(1.0, *self._flux_bounds))
        self._depth = _hat_projection((knots - log_min) / layout.step - origin, self._continuum - flux, count)
        offsets = np.arange(-layout.reach, layout.reach + 1) * layout.step
        kernel = np.exp(-0.5 * (offsets / layout.kernel_sigma_log)**2)
        kernel /= kernel.sum()
        self._convolved_depth = fftconvolve(self._depth, kernel, mode="valid")
        self._nodes = np.arange(layout.first, layout.last + 1) * layout.step      # log(lambda / lambda_min)
        # d(lambda) / lambda_min = exp(y) dy with y = log(lambda / lambda_min).
        self._weighted = CubicSpline(self._nodes, self._convolved_depth * np.exp(self._nodes), extrapolate=False)
        # Interval j holds c0 t^3 + c1 t^2 + c2 t + c3, t = y - node_j; its integral
        # from 0 to t is ((g0 t + g1) t + g2) t + g3) t with these coefficients.
        c = self._weighted.c
        self._integral_coefficients = np.stack([c[0] / 4, c[1] / 3, c[2] / 2, c[3]])
        self._interval_length = np.diff(self._nodes)               # exact differences, not quite `step`
        self._relative_edges = relative_edges
        self._flat_until, self._flat_level = self._flat_runs(knots, flux, kernel)

    def _flat_runs(self, knots, flux, kernel) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
        """Nodes whose whole dependency window holds one template value.

        Beyond the truncated LSF the convolved depth is exactly constant; the
        first variation is at most the outermost kernel weight times the
        template's, and the spline carries a variation k nodes on attenuated by
        ``SPLINE_COUPLING**k``.  ``_flat_margin`` more nodes bring the summed
        coupling below double-precision resolution.  A node is flat if the
        template's interpolant is constant within the LSF reach, the hat, that
        margin and one node for rounding of the knot positions; ``level`` is
        that constant (NaN otherwise) and ``until[i]`` the last node of the run
        of flat nodes at node ``i``'s level (-1 if node ``i`` is not flat).
        """
        layout = self._layout
        coupling = np.sum(SPLINE_COUPLING ** np.arange(kernel.size) * np.cumsum(kernel))
        self._flat_margin = max(0, int(np.ceil(np.log(np.finfo(float).epsneg / coupling) / np.log(SPLINE_COUPLING))))
        half = layout.reach + 2 + self._flat_margin
        x = (knots - layout.log_min) / layout.step                  # template knots in grid steps
        count = self._nodes.size
        varying = np.flatnonzero(flux[1:] != flux[:-1])             # intervals where the interpolant changes
        start = np.clip(np.floor(x[varying] - half).astype(int) - layout.first, 0, count)
        stop = np.clip(np.ceil(x[varying + 1] + half).astype(int) - layout.first + 1, 0, count)
        seen = np.cumsum(np.bincount(start, minlength=count + 1) - np.bincount(stop, minlength=count + 1))[:count] > 0
        level = np.where(seen, np.nan, np.interp(np.arange(layout.first, layout.last + 1), x, flux))
        same = ~seen[:-1] & ~seen[1:] & (level[:-1] == level[1:])
        ends = np.flatnonzero(~np.append(same, False))
        until = np.where(seen, -1, ends[np.searchsorted(ends, np.arange(count))])
        return until, level

    def _shift(self, rv_m_s: float) -> float:
        if not np.isfinite(rv_m_s) or abs(rv_m_s) > self.config.rv_padding_m_s:
            raise ValueError("RV [m/s] exceeds configured template padding.")
        return float(np.log(doppler_factor(rv_m_s)))

    def _pixel_integrals(self, shift: float) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.float64]]:
        """Each pixel's first spline interval, and the integrals of the weighted
        depth and of the weight over the pixel's rest-frame interval.

        A pixel spans ``oversample`` grid steps, so it touches at most
        ``oversample + 1`` spline intervals; each contributes its own integral
        between the pixel edges clipped to that interval.  Clipped at each
        interval's own length, the pieces tile one span exactly, over which the
        weight exp(y) integrates in closed form without cancellation.
        """
        step = self._layout.step
        lower = self._relative_edges[:-1] - shift
        upper = self._relative_edges[1:] - shift
        first = np.floor((lower - self._nodes[0]) / step).astype(int)
        interval = first[:, None] + np.arange(self.config.oversample + 1)
        left = self._nodes[interval]
        length = self._interval_length[interval]
        a = np.clip(lower[:, None] - left, 0.0, length)
        b = np.clip(upper[:, None] - left, 0.0, length)
        g0, g1, g2, g3 = self._integral_coefficients[:, interval]
        absorbed = ((((g0 * b + g1) * b + g2) * b + g3) * b - (((g0 * a + g1) * a + g2) * a + g3) * a).sum(axis=1)
        covered = self._nodes[first], self._nodes[first + self.config.oversample + 1]
        start, end = np.clip(lower, *covered), np.clip(upper, *covered)
        return first, absorbed, np.exp(start) * np.expm1(end - start)

    def __call__(self, rv_m_s: float) -> NDArray[np.float64]:
        """Return relative flux integrated over detector pixels at RV [m/s]."""
        first, absorbed, width = self._pixel_integrals(self._shift(rv_m_s))
        # A normalized positive LSF followed by a pixel average stays within
        # the template's range. Spline integration can overshoot at black troughs.
        flux = np.clip(self._continuum - absorbed / width, *self._flux_bounds)
        flat = self._flat_until[first] >= first + self.config.oversample + 1
        flux[flat] = self._flat_level[first[flat]]
        return flux

    def evaluate(self, rv_m_s: float) -> SpectrographOutput:
        """Return input, convolved and detector-sampled spectra at RV [m/s].

        ``input_flux`` and ``convolved_flux`` are sampled on the fixed grid
        ``highres_wavelength_um`` at the rest-frame position each sample has
        at this RV: the template as projected onto the internal grid (linearly
        interpolated), and the LSF-convolved spectrum the detector integration
        uses (the same spline).  Where both neighbouring nodes lie in one flat
        run, both are that run's level exactly, as the pixels are.
        """
        sampled, shift = self(rv_m_s), self._shift(rv_m_s)
        rest = self._grid - shift
        reach = self._layout.reach
        projected = np.interp(rest, self._nodes, self._depth[reach:self._depth.size - reach])
        convolved = self._weighted(rest) * np.exp(-rest)
        node = np.floor((rest - self._nodes[0]) / self._layout.step).astype(int)
        flat = self._flat_until[node] >= node + 1
        spectra = [np.clip(self._continuum - depth, *self._flux_bounds) for depth in (projected, convolved)]
        for spectrum in spectra:
            spectrum[flat] = self._flat_level[node[flat]]
        return SpectrographOutput(self.wavelength_um, self.pixel_width_um, sampled, self.highres_wavelength_um,
                                  *spectra)
