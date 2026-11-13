"""Synthetic, numerically motivated spectral/RV validation; no on-sky validation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_distribution
from scipy.integrate import trapezoid
from scipy.special import ndtr

from aophoton.detector import DetectorConfig, simulate_detector
from aophoton.radial_velocity import (
    CORRELATION_SCALE_SAMPLES, FISHER_DERIVATIVE_STEP_M_S, RVSearchBudgetError, recover_rv, rv_uncertainty,
    velocity_correlation_scale,
)
from aophoton.spectrograph import SpectrographConfig, SpectrographModel, template_support_um
from aophoton.spectrum import C_M_S, Spectrum, doppler_factor, load_spectrum, synthetic_spectrum


@pytest.fixture(scope="module")
def model():
    config = SpectrographConfig(wavelength_max_um=1.546)
    template = synthetic_spectrum(1.539, 1.547, samples=25_000, n_lines=50, seed=91)
    return SpectrographModel(template, config)


def test_continuum_pixel_flux_and_log_sampling():
    template = Spectrum(np.linspace(1.53, 1.58, 10), np.ones(10))
    config = SpectrographConfig()
    model = SpectrographModel(template, config)
    # Pixel integration in wavelength preserves a constant photon density.
    np.testing.assert_allclose(model(1234.5), 1, atol=4e-10, rtol=0)
    np.testing.assert_allclose(np.diff(np.log(model.wavelength_um)), model.dlog_pixel, rtol=1e-10)
    sampling = 1 / config.resolving_power / model.dlog_pixel
    assert config.pixels_per_resolution_element <= sampling < config.pixels_per_resolution_element * 1.001
    assert np.isclose(model.pixel_width_um.sum(), 0.03)
    detector = simulate_detector(model(0), model.pixel_width_um, DetectorConfig(),
                                 collecting_area_m2=10, coupling=0.4, throughput=0.15, noisy=False)
    expected = 2e7 * 10 * 300 * 0.4 * 0.15 * 0.03
    assert np.isclose(detector.expected_source_e.sum(), expected, rtol=1e-10)


def test_gaussian_lsf_width_and_pixel_broadening():
    config = SpectrographConfig(wavelength_min_um=1.54, wavelength_max_um=1.5406,
                               oversample=12)
    center = np.sqrt(config.wavelength_min_um * config.wavelength_max_um)
    wavelength = np.geomspace(1.539, 1.542, 80_000)
    intrinsic_sigma_m_s = 50.0
    flux = 1 - 0.8 * np.exp(-0.5 * (C_M_S * np.log(wavelength / center) / intrinsic_sigma_m_s)**2)
    model = SpectrographModel(Spectrum(wavelength, flux), config)
    output = model.evaluate(0)
    velocity = C_M_S * np.log(output.highres_wavelength_um / center)
    depth = 1 - output.convolved_flux
    centroid = np.sum(depth * velocity) / np.sum(depth)
    sigma_m_s = np.sqrt(np.sum(depth * (velocity - centroid)**2) / depth.sum())
    expected_lsf_sigma = C_M_S / config.resolving_power / np.sqrt(8 * np.log(2))
    # Variances of Gaussian intrinsic lines and Gaussian LSFs add. Grid projection
    # and 83 m/s LSF sampling contribute <0.3% width error, well below a pixel.
    assert sigma_m_s == pytest.approx(np.hypot(expected_lsf_sigma, intrinsic_sigma_m_s), rel=0.003)
    sampled_velocity = C_M_S * np.log(model.wavelength_um / center)
    sampled_depth = 1 - output.normalized_flux
    sampled_centroid = np.sum(sampled_depth * sampled_velocity) / np.sum(sampled_depth)
    sampled_sigma = np.sqrt(np.sum(sampled_depth * (sampled_velocity - sampled_centroid)**2) / sampled_depth.sum())
    pixel_width_m_s = C_M_S * model.dlog_pixel
    expected_sampled_sigma = np.sqrt(expected_lsf_sigma**2 + intrinsic_sigma_m_s**2 + pixel_width_m_s**2 / 12)
    # Integration adds the top-hat variance width^2/12; oversample=12 resolves it.
    assert sampled_sigma == pytest.approx(expected_sampled_sigma, rel=0.003)
    equivalent_width_in = trapezoid(1 - output.input_flux, output.highres_wavelength_um)
    equivalent_width_out = np.sum((1 - output.normalized_flux) * model.pixel_width_um)
    assert equivalent_width_out == pytest.approx(equivalent_width_in, rel=1e-7)



def _direct_pixel_means(template, config, rv_m_s, order=3, cutoff_sigma=9.0):
    """Independent oracle: pixel means by direct quadrature of the template.

    The template is linear in log(lambda) between its samples. Each sample
    interval is integrated with Gauss-Legendre nodes against the closed-form
    response of an untruncated Gaussian LSF integrated over a pixel in lambda:
    int_pixel G(l - s - u) exp(l) dl = exp(s + u + sigma^2/2) [Phi]_edges. No
    internal grid, projection, convolution kernel or spline is involved.
    """
    log_min, log_max = np.log([config.wavelength_min_um, config.wavelength_max_um])
    n_pixels = int(np.ceil((log_max - log_min) * config.resolving_power * config.pixels_per_resolution_element))
    edges = np.linspace(log_min, log_max, n_pixels + 1)
    sigma = 1 / (config.resolving_power * np.sqrt(8 * np.log(2)))
    shift = np.log(doppler_factor(rv_m_s))
    knots = np.log(template.wavelength_um)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = 0.5 * np.diff(knots)[:, None]
    points = 0.5 * (knots[:-1] + knots[1:])[:, None] + half * nodes
    mass = (half * weights * np.interp(points, knots, 1 - template.relative_flux) * np.exp(points)).ravel()
    points = points.ravel()
    below = np.concatenate([[0.0], np.cumsum(mass)])
    response = np.empty(edges.size)       # sum_q mass_q Phi((edge - shift - sigma^2 - u_q) / sigma)
    for index, edge in enumerate(edges):
        centre = edge - shift - sigma**2
        low, high = np.searchsorted(points, [centre - cutoff_sigma * sigma, centre + cutoff_sigma * sigma])
        response[index] = below[low] + mass[low:high] @ ndtr((centre - points[low:high]) / sigma)
    return 1 - np.exp(shift + sigma**2 / 2) * np.diff(response) / np.diff(np.exp(edges))


def test_unresolved_line_keeps_its_area_and_centroid_at_every_subpixel_shift():
    # A 35 m/s line is 7x narrower than the 248 m/s internal grid step. Point
    # sampling it made the area vary from 0.755 to 198 m/s with the shift; the
    # exact projection keeps the analytic area for every pixel and grid phase.
    config = SpectrographConfig(wavelength_min_um=1.55, wavelength_max_um=1.5506)
    center = np.sqrt(config.wavelength_min_um * config.wavelength_max_um)
    wavelength = np.geomspace(1.548, 1.553, 250_000)
    template = Spectrum(wavelength, 1 - 0.8 * np.exp(-0.5 * (C_M_S * np.log(wavelength / center) / 35.0)**2))
    model = SpectrographModel(template, config)
    step_m_s = C_M_S * model.dlog_pixel / config.oversample
    lsf_sigma_m_s = C_M_S * model.lsf_sigma_log
    analytic_m_s = 0.8 * np.sqrt(2 * np.pi) * 35.0
    velocity = C_M_S * np.log(model.wavelength_um / center)

    def moments(flux):
        absorbed = (1 - flux) * model.pixel_width_um
        centroid = absorbed @ velocity / absorbed.sum()
        return absorbed.sum() * C_M_S / center, centroid, absorbed @ (velocity - centroid)**2 / absorbed.sum()

    phases = [0, 0.1, 0.25, 1 / 3, 0.5, 0.71, 0.9]
    centroids, variance_ratios = [], []
    for rv in [0.0, 50.0, 125.0, 200.0, 250.0] + [(13 + phase) * step_m_s for phase in phases]:
        flux, direct = model(rv), _direct_pixel_means(template, config, rv)
        area, centroid, variance = moments(flux)
        # The Doppler factor stretches the line's width, and so its area, in lambda.
        assert area == pytest.approx(analytic_m_s * doppler_factor(rv), rel=1e-8)
        centroids.append(centroid - C_M_S * np.log(doppler_factor(rv)))
        variance_ratios.append(variance / moments(direct)[2])
        # A line narrower than the step gets a projection variance set by where
        # it falls between rest-frame nodes, not by the RV: the LSF variance is
        # off by at most step^2/6 (0.6% here), the profile by about 0.1%.
        assert np.max(np.abs(flux - direct)) < 2e-3 * np.max(1 - direct)
    assert np.all(np.abs(np.array(variance_ratios) - 1) < (step_m_s / lsf_sigma_m_s)**2 / 6)
    # The binned centroid carries a fixed 5 mm/s offset from weighting in
    # lambda. Neither it nor the width may depend on the shift.
    assert np.ptp(centroids) < 1e-4
    assert np.ptp(variance_ratios) < 1e-6


@pytest.mark.parametrize(("resolving_power", "rv_tolerance_m_s"), [(10_000, 0.05), (100_000, 2e-4)])
def test_default_fit_recovers_rv_from_direct_quadrature(resolving_power, rv_tolerance_m_s):
    # Seeded built-in template (600-1600 m/s lines) observed through the full
    # band. The observation comes from the grid-free oracle, so the fit tests
    # the default internal grid against the continuous model. Point sampling
    # the template at R = 10k gave 2324.65 m/s and a Fisher width 0.218x the
    # refined value.
    template = synthetic_spectrum(1.535, 1.575, seed=7)
    config = SpectrographConfig(resolving_power=resolving_power)
    injected = 2300.0
    observed = _direct_pixel_means(template, config, injected)
    variance = np.ones_like(observed)
    step = FISHER_DERIVATIVE_STEP_M_S
    slope = (_direct_pixel_means(template, config, injected + step)
             - _direct_pixel_means(template, config, injected - step)) / (2 * step)
    direct_sigma = np.sum(slope**2 / variance) ** -0.5
    errors = []
    for oversample in (4, 8, 16):
        model = SpectrographModel(template, replace(config, oversample=oversample))
        fit = recover_rv(model, observed, variance, search_min_m_s=-10_000, search_max_m_s=10_000)
        errors.append(abs(fit.rv_m_s - injected))
        assert rv_uncertainty(model, injected, variance) == pytest.approx(direct_sigma, rel=2e-5)
    # Default oversample meets the tolerance; refinement converges toward the oracle.
    assert errors[0] < rv_tolerance_m_s
    assert errors[2] < errors[0] / 50

@pytest.mark.parametrize(("low", "high", "resolving_power", "negative_control"),
                         [(1.0, 2.0, 30_000, True), (1.45, 1.80, 30_000, False), (1.54, 1.57, 100_000, False)])
def test_built_in_template_resolves_its_lines_on_wide_bands(low, high, resolving_power, negative_control):
    # The production template (simulation_template) fits a noiseless 2300 m/s
    # injection made through a template of the same seed and range refined to
    # 2-3x its density (>= 800k samples).  A fixed 50,000 samples spaced the
    # knots 4.2 km/s apart across 1-2 um, 7x the narrowest 600 m/s line, and
    # fitted 2329.93 m/s with +4.97% equivalent width.
    from aophoton.config import SimulationConfig
    from aophoton.simulation import simulation_template, synthetic_template_range_um
    from aophoton.spectrum import template_samples

    config = SimulationConfig()
    spectrograph = replace(config.spectrograph, wavelength_min_um=low, wavelength_max_um=high,
                           resolving_power=resolving_power)
    config = replace(config, spectrograph=spectrograph)
    a, b = synthetic_template_range_um(spectrograph)
    production = simulation_template(config)
    assert production.relative_flux.size == template_samples(a, b)
    seed = config.observation.template_seed
    references = [SpectrographModel(synthetic_spectrum(a, b, samples=max(n * production.relative_flux.size, m), seed=seed),
                                    spectrograph) for n, m in ((2, 800_000), (3, 1_200_000))]
    injected = 2300.0
    observations = [reference(injected) for reference in references]
    variance = np.ones_like(observations[0])
    model = SpectrographModel(production, spectrograph)

    def equivalent_width(flux):
        return np.dot(1 - flux, model.pixel_width_um)

    # The references agree with each other far below the acceptance targets.
    assert abs(equivalent_width(observations[0]) / equivalent_width(observations[1]) - 1) < 1e-9
    fits = [recover_rv(model, observed, variance, search_min_m_s=-5000, search_max_m_s=5000).rv_m_s
            for observed in observations]
    assert abs(fits[0] - fits[1]) < 1e-3
    assert abs(fits[1] - injected) < 0.1
    assert abs(equivalent_width(model(injected)) / equivalent_width(observations[1]) - 1) < 5e-3
    if negative_control:                     # the fixed 50,000 samples misses both targets by far
        coarse = SpectrographModel(synthetic_spectrum(a, b, samples=50_000, seed=seed), spectrograph)
        fit = recover_rv(coarse, observations[1], variance, search_min_m_s=-5000, search_max_m_s=5000)
        assert abs(fit.rv_m_s - injected) > 10
        assert equivalent_width(coarse(injected)) / equivalent_width(observations[1]) - 1 > 0.04


def test_detector_noise_budget_channels_and_coupling():
    config = DetectorConfig(exposure_s=10, continuum_photons_s_m2_um=1e6,
                            read_noise_e=3, dark_current_e_s=0.2,
                            background_e_s_pixel=0.4, extraction_pixels=5)
    output = simulate_detector([1, 0.5], [1e-5, 2e-5], config, collecting_area_m2=2,
                               coupling=[0.3, 0.15], throughput=0.25, n_channels=7, noisy=False)
    source = np.array([15.0, 7.5])
    np.testing.assert_allclose(output.expected_source_e, source)
    assert output.expected_dark_e == 70
    assert output.expected_background_e == 140
    assert output.read_variance_e2 == 315
    np.testing.assert_allclose(output.variance_e2, source + 70 + 140 + 315)
    np.testing.assert_allclose(output.flux_e, source)
    np.testing.assert_allclose(output.snr, source / np.sqrt(source + 525))
    single = simulate_detector([1, 0.5], [1e-5, 2e-5], config, collecting_area_m2=2,
                               coupling=[0.3, 0.15], throughput=0.25, n_channels=1, noisy=False)
    np.testing.assert_allclose(single.expected_source_e, output.expected_source_e)
    np.testing.assert_allclose(output.variance_e2 - source, 7 * (single.variance_e2 - source))


def test_detector_empirical_mean_variance_and_seed():
    count = 100_000
    config = DetectorConfig(exposure_s=10, continuum_photons_s_m2_um=1e6,
                            read_noise_e=3, dark_current_e_s=0.2,
                            background_e_s_pixel=0.4, extraction_pixels=3)
    args = dict(normalized_flux=np.ones(count), pixel_width_um=np.full(count, 1e-5),
                config=config, collecting_area_m2=2, coupling=0.3, throughput=0.25, n_channels=2)
    output = simulate_detector(**args, rng=np.random.default_rng(142))
    duplicate = simulate_detector(**args, rng=np.random.default_rng(142))
    np.testing.assert_array_equal(output.flux_e, duplicate.flux_e)
    expected_variance = output.variance_e2[0]
    # Four standard errors for sample mean and approximately Gaussian variance
    # with 100,000 independent draws tests both subtraction and every noise term.
    assert abs(output.flux_e.mean() - output.expected_source_e[0]) < 4 * np.sqrt(expected_variance / count)
    assert abs(output.flux_e.var(ddof=1) / expected_variance - 1) < 4 * np.sqrt(2 / (count - 1))


def test_zero_noise_non_grid_rv(model):
    injected = 1234.56789
    expected = simulate_detector(model(injected), model.pixel_width_um, DetectorConfig(),
                                 collecting_area_m2=10, coupling=0.3, throughput=0.15, noisy=False)
    recovered = recover_rv(model, expected.flux_e, expected.variance_e2,
                           continuum_e=expected.continuum_e)
    # Same forward operator gives a strict numerical test: <0.01 m/s, compared
    # with ~1000 m/s per pixel and a 250 m/s coarse RV grid. No noise tolerance.
    assert abs(recovered.rv_m_s - injected) < 0.01
    assert recovered.minimum_chi2 < 1e-8
    assert not recovered.at_boundary
    assert recovered.uncertainty_m_s > 0


def test_rv_monte_carlo_uncertainty_calibration(model):
    injected = 237.123
    rng = np.random.default_rng(44)
    config = DetectorConfig()
    residuals = []
    for _ in range(64):
        observation = simulate_detector(model(injected), model.pixel_width_um, config,
                                         collecting_area_m2=10, coupling=0.3, throughput=0.15, rng=rng)
        estimate = recover_rv(model, observation.flux_e, observation.variance_e2,
                               continuum_e=observation.continuum_e,
                               search_min_m_s=-1000, search_max_m_s=1500, grid_size=21)
        assert not estimate.at_boundary
        residuals.append((estimate.rv_m_s - injected) / estimate.uncertainty_m_s)
    residuals = np.asarray(residuals)
    # Analytic 3-sigma sampling bounds: a calibrated Fisher uncertainty should
    # give unit-normal standardized RV errors over 64 independent observations.
    assert abs(residuals.mean()) < 3 / np.sqrt(residuals.size)
    statistic = (residuals.size - 1) * residuals.var(ddof=1)
    lower, upper = chi2_distribution.ppf([0.00135, 0.99865], residuals.size - 1)
    assert lower < statistic < upper


def test_rv_precision_improves_with_photons_and_channel_cost(model):
    forecasts = []
    for coupling, channels in [(0.1, 1), (0.4, 1), (0.4, 19)]:
        observation = simulate_detector(model(0), model.pixel_width_um, DetectorConfig(),
                                         collecting_area_m2=10, coupling=coupling,
                                         throughput=0.15, n_channels=channels, noisy=False)
        forecasts.append(rv_uncertainty(model, 0, observation.variance_e2, observation.continuum_e))
    assert forecasts[1] < forecasts[0]
    assert forecasts[2] > forecasts[1]


def test_rv_search_boundary_is_reported(model):
    result = recover_rv(model, model(2500), np.full(model.wavelength_um.size, 1e-4),
                        search_min_m_s=-1000, search_max_m_s=1000, grid_size=11)
    assert result.at_boundary
    assert result.rv_m_s == 1000


def test_wide_rv_search_cannot_step_over_true_basin():
    # +/-400 km/s with the default 81-node grid would space nodes ~10 km/s apart,
    # wider than the R=100,000 chi-square basin, and lock onto a precise but wrong
    # alias. The grid must densify to the template's velocity correlation scale.
    config = SpectrographConfig(rv_padding_m_s=900_000.0)
    template = synthetic_spectrum(config.wavelength_min_um - 0.005, config.wavelength_max_um + 0.005, seed=7)
    wide = SpectrographModel(template, config)
    variance = np.full(wide.wavelength_um.size, 1e-4)
    result = recover_rv(wide, wide(5000.0), variance, search_min_m_s=-400_000, search_max_m_s=400_000)
    assert abs(result.rv_m_s - 5000.0) < 0.01
    assert result.grid_velocities_m_s.size > 81
    assert result.reduced_chi2 < 1e-8


def test_rv_search_narrower_than_two_derivative_steps():
    # A +/-0.25 m/s search with 1.25 m/s template padding is a valid configuration
    # (search + 1 m/s derivative step fits). Correlation-scale probes must stay
    # inside the search instead of crossing over and leaving template coverage.
    config = SpectrographConfig(wavelength_max_um=1.546, rv_padding_m_s=1.25)
    template = synthetic_spectrum(1.539, 1.547, samples=25_000, n_lines=50, seed=91)
    tight = SpectrographModel(template, config)
    variance = np.full(tight.wavelength_um.size, 1e-4)
    result = recover_rv(tight, tight(0.1), variance, search_min_m_s=-0.25, search_max_m_s=0.25)
    assert abs(result.rv_m_s - 0.1) < 0.01
    assert result.reduced_chi2 < 1e-8


def test_reduced_chi2_flags_a_mismatched_template(model):
    # A wrong template leaves residuals far above the stated variance.
    variance = np.full(model.wavelength_um.size, 1e-4)
    observed = model(0.0) + 0.05 * np.sin(np.arange(variance.size))
    result = recover_rv(model, observed, variance)
    # Far outside the ~1 +/- sqrt(2/dof) range of a correctly described spectrum.
    assert result.reduced_chi2 > 1 + 10 * np.sqrt(2 / (variance.size - 1))


@pytest.mark.parametrize("level", [0.0, 0.5, 2.0])
def test_constant_templates_preserve_flux_and_have_no_doppler_information(level):
    wave = np.geomspace(1.53, 1.58, 100)
    model = SpectrographModel(Spectrum(wave, np.full_like(wave, level)))
    for rv in (-1234.5, 2300.0):
        output = model.evaluate(rv)
        for flux in (output.normalized_flux, output.input_flux, output.convolved_flux):
            np.testing.assert_array_equal(flux, np.full_like(flux, level))
        assert np.isinf(rv_uncertainty(model, rv, np.ones_like(output.normalized_flux)))


def _plateau(config, level=0.5, remote=1.0, wavelength_um=None):
    """``level`` at every template sample the model can read (those inside
    ``template_support_um`` and the two bracketing it), ``remote`` elsewhere."""
    low, high = template_support_um(config)
    if wavelength_um is None:
        wavelength_um = np.unique(np.concatenate([np.geomspace(1.53, 1.58, 20_001), [low, high]]))
    flux = np.where((wavelength_um >= low) & (wavelength_um <= high), level, remote)
    return wavelength_um, flux, (wavelength_um < low) | (wavelength_um > high)


REPRESENTATIVE_RVS = (-29_000.0, -5_000.0, -1_234.5, 0.0, 2_300.0, 29_000.0)


@pytest.mark.parametrize("fixture", ["exact_support", "review_plateau"])
def test_flat_reachable_template_has_no_doppler_information(fixture):
    # 0.5 everywhere the model can read and 1.0 only beyond: the reachable
    # spectrum is flat, so no pixel can depend on RV.  Referencing depth to the
    # whole template's range gave a 0.5 depth, ~3e-11 roundoff that moved with
    # RV, and a fitted "RV" of 8600.68 +/- 1.66e11 m/s from flat data.
    config = SpectrographConfig()
    if fixture == "exact_support":
        wavelength, flux, remote = _plateau(config)
    else:                                   # the review's plateau, 1.538-1.572 um inside 1.53-1.58 um
        wavelength = np.geomspace(1.53, 1.58, 20_000)
        flux = np.where((wavelength > 1.538) & (wavelength < 1.572), 0.5, 1.0)
        remote = flux == 1.0
    observed = np.full(SpectrographModel(Spectrum(wavelength, flux), config).wavelength_um.size, 0.5)
    variance = np.ones_like(observed)
    rng = np.random.default_rng(3)
    altered = flux.copy()
    altered[remote] = rng.uniform(0.0, 3.0, remote.sum())       # only samples the model cannot reach
    for template_flux in (flux, altered):
        model = SpectrographModel(Spectrum(wavelength, template_flux), config)
        for rv in REPRESENTATIVE_RVS:
            output = model.evaluate(rv)
            np.testing.assert_array_equal(output.normalized_flux, observed)
            np.testing.assert_array_equal(output.convolved_flux, np.full_like(output.convolved_flux, 0.5))
            np.testing.assert_array_equal(output.input_flux, np.full_like(output.input_flux, 0.5))
            assert np.isinf(rv_uncertainty(model, rv, variance))
            assert np.isnan(velocity_correlation_scale(model, rv, variance))
        with pytest.raises(ValueError, match="no RV information"):
            recover_rv(model, observed, variance)


def test_weak_line_on_a_flat_plateau_keeps_its_rv_information():
    # Positive control: the same 0.5 plateau with a 0.2%-deep, 2 km/s line in
    # the band.  It must keep finite information and a correct fit, so the flat
    # case is not handled by a threshold that would also erase shallow lines.
    config = SpectrographConfig()
    wavelength, flux, remote = _plateau(config)
    line = 1 - 0.002 * np.exp(-0.5 * (C_M_S * np.log(wavelength / 1.555) / 2_000.0)**2)
    model = SpectrographModel(Spectrum(wavelength, np.where(remote, 1.0, 0.5 * line)), config)
    variance = np.full(model.wavelength_um.size, 1e-14)        # SNR 1e7 on a 0.001-deep line: sigma ~0.2 m/s
    sigma = rv_uncertainty(model, 2300.0, variance)
    assert np.isfinite(sigma) and sigma < 1.0
    fit = recover_rv(model, model(2300.0), variance, search_min_m_s=-10_000, search_max_m_s=10_000)
    assert fit.rv_m_s == pytest.approx(2300.0, abs=1e-3)
    assert not fit.at_boundary


def _padded_feature(low, high, level=0.5, peak=1.0, rest_rv_m_s=24_000.0, half_width_m_s=600.0):
    """The acceptance review's template: ``level`` with one triangular feature
    peaking at ``peak``, centred where an RV of ``rest_rv_m_s`` shifts it onto
    the band's blue edge; 120,000 log-spaced samples over 0.1% beyond the band."""
    wavelength = np.geomspace(low * 0.999, high * 1.001, 120_000)
    offset = C_M_S * np.log(wavelength / (low / doppler_factor(rest_rv_m_s)))
    flux = level + (peak - level) * np.maximum(1 - np.abs(offset) / half_width_m_s, 0.0)
    return wavelength, flux


SEARCH_RVS = (-10_000.0, -6_000.0, -1_234.5, 0.0, 2_300.0, 6_000.0, 10_000.0)


@pytest.mark.parametrize(("low", "high"), [(1.54, 1.57), (1.555, 1.55505)])
@pytest.mark.parametrize(("level", "peak"), [(0.5, 1.0), (1.0, 0.5)])
def test_feature_outside_the_search_adds_no_information_at_any_padding(low, high, level, peak):
    # The feature is 24 km/s blueward of the band: at the nearest searched RV
    # (+10 km/s, +1 m/s for the derivative) it is still >13 km/s outside, beyond
    # the 7.7 km/s truncated LSF, so no pixel depends on RV anywhere in the
    # search.  Widening the padding from 11 to 30 km/s brings the feature into
    # the model's support.  It used to raise the depth reference to the
    # feature, leave the plateau 0.49 deep, and return ~3e-11 roundoff that moved
    # with RV: a "fit" of -7204 m/s with finite sigma from exactly flat data.
    wavelength, flux = _padded_feature(low, high, level, peak)
    for padding in (11_000.0, 30_000.0):
        model = SpectrographModel(Spectrum(wavelength, flux),
                                  SpectrographConfig(wavelength_min_um=low, wavelength_max_um=high,
                                                     rv_padding_m_s=padding))
        observed = np.full(model.wavelength_um.size, level)
        variance = np.ones_like(observed)
        for rv in SEARCH_RVS:
            for probe in (rv - 1.0, rv, rv + 1.0):              # the Fisher derivative's evaluations
                np.testing.assert_array_equal(model(probe), observed)
            assert np.isinf(rv_uncertainty(model, rv, variance))
            assert np.isnan(velocity_correlation_scale(model, rv, variance))
        with pytest.raises(ValueError, match="no RV information"):
            recover_rv(model, observed, variance, search_min_m_s=-10_000, search_max_m_s=10_000)


def test_padded_feature_becomes_observable_inside_the_band():
    # Positive control for the local-support rule: the same feature is seen,
    # and its RV recovered, as soon as a searched RV brings it into the band.
    wavelength, flux = _padded_feature(1.54, 1.57)
    model = SpectrographModel(Spectrum(wavelength, flux), SpectrographConfig(rv_padding_m_s=30_000.0))
    observed = model(27_000.0)
    assert np.abs(observed - 0.5).max() > 0.08
    variance = np.full(observed.size, 1e-8)
    assert np.isfinite(rv_uncertainty(model, 27_000.0, variance))
    fit = recover_rv(model, observed, variance, search_min_m_s=20_000, search_max_m_s=29_000)
    assert fit.rv_m_s == pytest.approx(27_000.0, abs=1e-3)


def test_weak_line_beside_a_padded_feature_is_independent_of_the_padding():
    # A 0.2%-deep in-band line on the 0.5 plateau, with the padded feature
    # present: its information and fit survive, and widening the padding (which
    # changes the depth reference from 0.5 to the feature's 0.99) changes the
    # spectrum by roundoff only.  Before the width was integrated without
    # cancellation that difference was ~3e-11; a few roundings of depth-sized
    # quantities now bound it, allowed here 16 ulp of the plateau.
    wavelength, flux = _padded_feature(1.54, 1.57)
    line = 1 - 0.002 * np.exp(-0.5 * (C_M_S * np.log(wavelength / 1.555) / 2_000.0)**2)
    template = Spectrum(wavelength, flux - 0.5 + 0.5 * line)
    models = [SpectrographModel(template, SpectrographConfig(rv_padding_m_s=padding)) for padding in (11_000.0, 30_000.0)]
    for rv in SEARCH_RVS:
        np.testing.assert_allclose(models[0](rv), models[1](rv), rtol=0, atol=16 * np.spacing(0.5))
    variance = np.full(models[1].wavelength_um.size, 1e-14)
    for model in models:
        assert rv_uncertainty(model, 2300.0, variance) < 1.0
        fit = recover_rv(model, model(2300.0), variance)
        assert fit.rv_m_s == pytest.approx(2300.0, abs=1e-3) and not fit.at_boundary


BANDS = [(1.54, 1.57), (0.4, 0.5), (0.5, 0.7), (0.6, 0.8), (1.0, 2.0), (2.0, 2.1), (1.5, 1.6), (1.55, 1.56)]


@pytest.mark.parametrize(("low", "high"), BANDS)
@pytest.mark.parametrize("resolving_power", [3_000, 10_000, 30_000, 100_000])
def test_advertised_support_is_accepted_and_any_physical_trim_is_not(low, high, resolving_power):
    # A template whose end samples are exactly template_support_um is accepted;
    # 16 of these 32 were refused when the helper's exp() and the constructor's
    # log() disagreed in the last bit.  Trimming either end by 1 m/s of
    # velocity, far below any line width yet 1e5 ulp, is refused.
    config = SpectrographConfig(wavelength_min_um=low, wavelength_max_um=high, resolving_power=resolving_power)
    a, b = template_support_um(config)
    SpectrographModel(Spectrum(np.linspace(a, b, 10), np.ones(10)), config)
    trim = doppler_factor(1.0)
    for trimmed in ((a * trim, b), (a, b / trim)):
        with pytest.raises(ValueError, match="Template must cover"):
            SpectrographModel(Spectrum(np.linspace(*trimmed, 10), np.ones(10)), config)


def test_public_highres_grid_is_fixed_in_the_observer_frame():
    # The default grid of the reviewed release: 23,153 samples from 1.54 to
    # 1.57 um.  Every RV returns that grid, so spectra stack and difference
    # sample by sample, and a line moves along it in the Doppler direction.
    model = SpectrographModel(synthetic_spectrum())
    grid = model.highres_wavelength_um
    assert grid.size == 23_153
    assert grid[0] == pytest.approx(1.54, rel=1e-15) and grid[-1] == pytest.approx(1.57, rel=1e-15)
    step = model.dlog_pixel / model.config.oversample
    np.testing.assert_allclose(np.diff(np.log(grid)), step, rtol=1e-6)
    outputs = [model.evaluate(rv) for rv in (0.0, 2300.0, -5000.0)]
    for output in outputs:
        np.testing.assert_array_equal(output.highres_wavelength_um, grid)
        assert output.input_flux.shape == output.convolved_flux.shape == grid.shape
        np.testing.assert_array_equal(output.normalized_flux, model(0.0) if output is outputs[0] else
                                      model(2300.0) if output is outputs[1] else model(-5000.0))
    assert np.stack([output.convolved_flux for output in outputs]).shape == (3, grid.size)
    # One isolated line at 1.555 um: its minimum moves to longer wavelength for
    # positive RV, by lambda (D - 1), to within one grid step.
    wavelength = np.geomspace(1.53, 1.58, 200_001)
    line = Spectrum(wavelength, 1 - 0.6 * np.exp(-0.5 * (C_M_S * np.log(wavelength / 1.555) / 1_500.0)**2))
    single = SpectrographModel(line)
    for rv in (0.0, 2300.0, -5000.0):
        output = single.evaluate(rv)
        for flux in (output.input_flux, output.convolved_flux):
            moved = grid[np.argmin(flux)]
            assert abs(np.log(moved / (1.555 * doppler_factor(rv)))) <= step
        # The diagnostic spectrum, integrated over each pixel by the trapezoid
        # rule on the grid, reproduces the production pixel means to within the
        # rule's own error bound, h^2/12 max|f''| per unit width.
        x = np.log(grid)
        flux = output.convolved_flux
        weights = np.exp(x)
        integrand = flux * weights
        per_interval = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(x)
        trapezoid_means = per_interval.reshape(-1, single.config.oversample).sum(axis=1) \
            / np.diff(np.exp(np.log(single.pixel_edges_um)))
        bound = np.max(np.abs(np.diff(integrand, 2))) / 12 / weights.min() * 1.5
        assert np.max(np.abs(trapezoid_means - output.normalized_flux)) < bound


def test_saturated_template_can_be_observed_by_detector():
    wave = np.geomspace(1.53, 1.58, 4001)
    template = Spectrum(wave, np.where((wave > 1.545) & (wave < 1.55), 0.0, 1.0))
    model = SpectrographModel(template)
    flux = model(2300.0)
    # The trough's interior is many LSF widths from either transition.
    assert np.max(flux[(model.wavelength_um > 1.546) & (model.wavelength_um < 1.549)]) < 1e-9
    assert np.all((flux >= 0.0) & (flux <= 1.0))
    detected = simulate_detector(flux, model.pixel_width_um, DetectorConfig(), collecting_area_m2=10,
                                 coupling=0.4, throughput=0.15, noisy=False)
    np.testing.assert_allclose(detected.expected_source_e, detected.continuum_e * flux)


def test_unidentifiable_rv_is_rejected(model):
    # Neither a zero-photon continuum nor a featureless template constrains RV;
    # returning an arbitrary optimizer coordinate would misrepresent a detection.
    flux = np.zeros_like(model.wavelength_um)
    variance = np.ones_like(flux)
    with pytest.raises(ValueError, match="no RV information"):
        recover_rv(model, flux, variance, continuum_e=0, grid_size=5)
    with pytest.raises(ValueError, match="no RV information"):
        recover_rv(lambda rv: np.ones(10), np.ones(10), np.ones(10), grid_size=5)


def test_template_seed_validation_coverage_and_shift(tmp_path):
    a = synthetic_spectrum(seed=17)
    b = synthetic_spectrum(seed=17)
    np.testing.assert_array_equal(a.relative_flux, b.relative_flux)
    assert not np.array_equal(a.relative_flux, synthetic_spectrum(seed=18).relative_flux)
    assert doppler_factor(1000) > 1
    assert doppler_factor(1000) * doppler_factor(-1000) == pytest.approx(1)
    for wavelength, flux in [([1, 1, 2], [1, 1, 1]), ([1, 2, 3], [1, np.nan, 1]),
                             ([1, 2, 3], [1, -1, 1]), ([1, 3, 2], [1, 1, 1])]:
        with pytest.raises(ValueError):
            Spectrum(np.asarray(wavelength), np.asarray(flux))
    path = tmp_path / "template.txt"
    np.savetxt(path, np.column_stack([a.wavelength_um, a.relative_flux]))
    imported = load_spectrum(path)
    np.testing.assert_array_equal(imported.relative_flux, a.relative_flux)
    short = Spectrum(np.linspace(1.54, 1.57, 100), np.ones(100))
    with pytest.raises(ValueError, match="padding"):
        SpectrographModel(short)
    with pytest.raises(ValueError, match="padding"):
        SpectrographModel(a)(30_001)


def test_fits_import_explicit_units_and_schema(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "template.fits"
    wavelength_nm = np.linspace(1535, 1575, 100)
    flux = np.linspace(0.5, 1, 100)
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name="WAVELENGTH", format="D", unit="nm", array=wavelength_nm),
        fits.Column(name="FLUX", format="D", array=flux),
    ])
    hdu.writeto(path)
    imported = load_spectrum(path)
    np.testing.assert_allclose(imported.wavelength_um, wavelength_nm / 1000)
    np.testing.assert_array_equal(imported.relative_flux, flux)
    hdu.columns["FLUX"].unit = "%"
    hdu.writeto(path, overwrite=True)
    np.testing.assert_allclose(load_spectrum(path).relative_flux, flux / 100)
    hdu.columns["FLUX"].unit = "Jy"
    hdu.writeto(path, overwrite=True)
    with pytest.raises(ValueError, match="dimensionless"):
        load_spectrum(path)
    hdu.columns["FLUX"].unit = None
    hdu.columns["WAVELENGTH"].unit = None
    hdu.writeto(path, overwrite=True)
    with pytest.raises(ValueError, match="unit"):
        load_spectrum(path)
    primary = fits.PrimaryHDU(flux)
    primary.header.update({"CTYPE1": "WAVE", "CUNIT1": "nm", "CRVAL1": 1535,
                           "CDELT1": 40 / 99, "CRPIX1": 1})
    primary.writeto(path, overwrite=True)
    np.testing.assert_allclose(load_spectrum(path).wavelength_um, wavelength_nm / 1000)
    primary.header["CTYPE1"] = "WAVE-LOG"
    primary.writeto(path, overwrite=True)
    with pytest.raises(ValueError, match="linear"):
        load_spectrum(path)


def _primary_wave_fits(path, header):
    fits = pytest.importorskip("astropy.io.fits")
    primary = fits.PrimaryHDU(np.array([1.0, 0.9, 1.0, 0.95]))
    primary.header.update(header)
    primary.writeto(path, overwrite=True)
    return primary.header


@pytest.mark.parametrize("header", [
    # The audit case: PC1_1 scales CDELT1, so the step is 2 nm, not 1 nm.
    {"CRPIX1": 1.0, "CRVAL1": 1500.0, "CDELT1": 1.0, "PC1_1": 2.0, "CUNIT1": "nm"},
    {"CRPIX1": 2.5, "CRVAL1": 1.5, "CDELT1": 0.0005, "PC1_1": 4.0, "CUNIT1": "um"},
    {"CRPIX1": 1.0, "CRVAL1": 15000.0, "CD1_1": 30.0, "CUNIT1": "Angstrom"},
    # CD1_1 overrides CDELT1; PC1_1 alone implies CDELT1 = 1.
    {"CRPIX1": 3.0, "CRVAL1": 1500.0, "CD1_1": 3.0, "CDELT1": 5.0, "CUNIT1": "nm"},
    {"CRPIX1": 1.0, "CRVAL1": 1500.0, "PC1_1": 2.0, "CUNIT1": "nm", "WCSAXES": 1},
])
def test_fits_linear_wave_axis_matches_astropy_wcs(tmp_path, header):
    wcs = pytest.importorskip("astropy.wcs")
    units = pytest.importorskip("astropy.units")
    path = tmp_path / "primary.fits"
    written = _primary_wave_fits(path, {"CTYPE1": "WAVE", **header})
    reference = wcs.WCS(written)
    expected = (reference.pixel_to_world_values(np.arange(4)) * units.Unit(reference.wcs.cunit[0])).to_value(units.um)
    np.testing.assert_allclose(load_spectrum(path).wavelength_um, expected, rtol=1e-13, atol=0)
    if header.get("PC1_1") == 2.0 and header.get("CDELT1") == 1.0:
        np.testing.assert_allclose(load_spectrum(path).wavelength_um[:3], [1.500, 1.502, 1.504], rtol=1e-13)


@pytest.mark.parametrize("alternate", [{"PC1_1A": 2.0}, {"CD1_1A": 2.0}])
def test_alternate_wcs_does_not_block_the_primary_wave_axis(tmp_path, alternate):
    # A linear primary WAVE axis (1500-1504 nm, CDELT1 only) next to a legal
    # alternate description A (1600-1608 nm through PC1_1A or CD1_1A).  Astropy
    # reads both independently; the loader returns the primary in um.
    fits = pytest.importorskip("astropy.io.fits")
    wcs = pytest.importorskip("astropy.wcs")
    primary = fits.PrimaryHDU(np.ones(5))
    primary.header.update({"CTYPE1": "WAVE", "CUNIT1": "nm", "CRVAL1": 1500.0, "CRPIX1": 1.0, "CDELT1": 1.0,
                           "CTYPE1A": "WAVE", "CUNIT1A": "nm", "CRVAL1A": 1600.0, "CRPIX1A": 1.0,
                           **({"CDELT1A": 1.0} if "PC1_1A" in alternate else {}), **alternate})
    path = tmp_path / "alternate.fits"
    primary.writeto(path)
    pixels = np.arange(5.0)
    for key, expected_nm in ((" ", [1500, 1501, 1502, 1503, 1504]), ("A", [1600, 1602, 1604, 1606, 1608])):
        solution = wcs.WCS(primary.header, key=key)
        np.testing.assert_allclose(solution.pixel_to_world_values(pixels) * 1e9, expected_nm, rtol=1e-13)
    np.testing.assert_allclose(load_spectrum(path).wavelength_um, [1.500, 1.501, 1.502, 1.503, 1.504], rtol=1e-13)
    # The primary axis is still held to the linear model.
    for unsupported in ({"PC1_2": 0.1}, {"PV1_1": 1.0}, {"DP1.EXTRA": 1.0}, {"D2IMDIS1": "Lookup"}):
        header = primary.header.copy()
        header.update(unsupported)
        fits.PrimaryHDU(np.ones(5), header=header).writeto(path, overwrite=True)
        with pytest.raises(ValueError, match="not supported for a linear 1D WAVE axis"):
            load_spectrum(path)


@pytest.mark.parametrize(("extra", "message"), [
    ({"CTYPE1": "WAVE-TAB"}, "linear"),
    ({"CTYPE1": "AWAV"}, "air wavelengths"),
    ({"CTYPE1": "FREQ", "CUNIT1": "Hz"}, "CTYPE1='FREQ'"),
    ({"CD1_1": 2.0, "PC1_1": 2.0}, "both CD1_1 and PC1_1"),
    ({"WCSAXES": 2}, "WCSAXES=2"),
    ({"PC1_2": 0.1}, "PC1_2"),
    ({"PV1_1": 1.0}, "PV1_1"),
    ({"CPDIS1": "Lookup"}, "CPDIS1"),
    ({"CUNIT1": "s"}, "not a length"),
    ({"CDELT1": 0.0}, "nonzero"),
])
def test_fits_wcs_outside_the_linear_model_is_refused(tmp_path, extra, message):
    path = tmp_path / "primary.fits"
    header = {"CTYPE1": "WAVE", "CRPIX1": 1.0, "CRVAL1": 1500.0, "CDELT1": 1.0, "CUNIT1": "nm"}
    _primary_wave_fits(path, {**header, **extra})
    with pytest.raises(ValueError, match=message):
        load_spectrum(path)


@pytest.mark.parametrize("field,value", [("exposure_s", 0), ("read_noise_e", -1),
                                         ("dark_current_e_s", np.nan), ("extraction_pixels", 1.5)])
def test_invalid_detector_config(field, value):
    with pytest.raises(ValueError):
        replace(DetectorConfig(), **{field: value})


@pytest.mark.parametrize("field,value", [("resolving_power", 0), ("throughput", 1.1),
                                         ("oversample", 1), ("rv_padding_m_s", np.nan)])
def test_invalid_spectrograph_config(field, value):
    with pytest.raises(ValueError):
        replace(SpectrographConfig(), **{field: value})


def test_correlation_scale_is_not_taken_from_a_featureless_midpoint():
    # Narrow lines that lie in band only away from the search midpoint: a scale
    # sampled at the midpoint alone sees a featureless model and keeps the
    # coarse 81-node grid (~10 km/s spacing) that steps over the true basin.
    config = SpectrographConfig(rv_padding_m_s=900_000.0)
    wavelength = np.exp(np.linspace(np.log(1.54 / 1.004), np.log(1.57 * 1.004), 200_000))
    rng = np.random.default_rng(2)
    flux = np.ones_like(wavelength)
    for centre, depth, width in zip(rng.uniform(np.log(1.5686), np.log(1.5698), 25),
                                    rng.uniform(0.08, 0.65, 25), rng.uniform(600, 1600, 25) / C_M_S):
        flux *= 1 - depth * np.exp(-0.5 * ((np.log(wavelength) - centre) / width) ** 2)
    edge_lines = SpectrographModel(Spectrum(wavelength, flux), config)
    variance = np.full(edge_lines.wavelength_um.size, 1e-4)
    result = recover_rv(edge_lines, edge_lines(5123.4), variance, search_min_m_s=-100_000, search_max_m_s=700_000)
    assert abs(result.rv_m_s - 5123.4) < 0.01
    assert result.reduced_chi2 < 1e-8


def test_line_entering_a_narrow_band_between_fixed_probes_is_found():
    # A 9-pixel band (~5.8 km/s wide) searched over +/-400 km/s: the only line
    # is in band for a few km/s of RV, between nine evenly spaced probes. Probes
    # spaced by half the band's velocity width must still see it.
    from aophoton.spectrum import doppler_factor

    injected = 55_123.4
    wavelength = np.geomspace(1.549, 1.561, 200_000)
    rest_centre = 1.555015 / doppler_factor(injected)
    flux = 1 - 0.65 * np.exp(-0.5 * (C_M_S * np.log(wavelength / rest_centre) / 100) ** 2)
    narrow = SpectrographModel(Spectrum(wavelength, flux), SpectrographConfig(
        wavelength_min_um=1.555, wavelength_max_um=1.55503, resolving_power=150_000, rv_padding_m_s=900_000))
    variance = np.full(narrow.wavelength_um.size, 1e-4)
    result = recover_rv(narrow, narrow(injected), variance, search_min_m_s=-400_000, search_max_m_s=400_000)
    assert abs(result.rv_m_s - injected) < 0.01
    assert result.reduced_chi2 < 1e-8


class _CountedModel:
    """Model wrapper that counts calls and keeps the pixel edges the search reads."""

    def __init__(self, model):
        self.model, self.calls = model, 0
        self.pixel_edges_um = getattr(model, "pixel_edges_um", None)

    def __call__(self, rv_m_s):
        self.calls += 1
        return self.model(rv_m_s)


def test_symmetric_two_pixel_line_search_is_bounded():
    # A 600 m/s line centred between the only two pixels: the sampled spectrum
    # has almost no contrast at RV 0 but still moves with RV. Contrast/gradient
    # alone gave a 0.0002 m/s scale and a ~2e8-node grid for +/-10 km/s.
    low, high = 1.555, 1.555009
    centre = np.sqrt(low * high)
    wavelength = centre * np.exp(np.linspace(-2e-4, 2e-4, 100_001))
    template = Spectrum(wavelength, 1 - 0.8 * np.exp(-0.5 * (C_M_S * np.log(wavelength / centre) / 600) ** 2))
    model = SpectrographModel(template, SpectrographConfig(wavelength_min_um=low, wavelength_max_um=high))
    assert model.wavelength_um.size == 2
    variance = np.ones(2)
    assert velocity_correlation_scale(model, 0.0, variance) > 100
    for injected in (123.4, -2345.6):
        counted = _CountedModel(model)
        result = recover_rv(counted, model(injected), variance)
        assert abs(result.rv_m_s - injected) < 1e-4
        assert result.model_evaluations == counted.calls < 1_000
        assert result.reduced_chi2 < 1e-12


def test_unresolvable_rv_search_fails_before_allocating_its_grid():
    # A 20 m/s periodic model has a 3.3 m/s correlation scale; resolving +/-300
    # km/s would take ~3.6e5 nodes. The search must stop after its scale probes.
    phases = np.random.default_rng(1).uniform(0, 2 * np.pi, 50)
    counted = _CountedModel(lambda rv: 1 + 0.1 * np.sin(2 * np.pi * rv / 20 + phases))
    observed, variance = counted.model(3.3), np.ones(50)
    with pytest.raises(RVSearchBudgetError, match="Narrow the search interval") as failure:
        recover_rv(counted, observed, variance, search_min_m_s=-300_000, search_max_m_s=300_000)
    assert "3.327 m/s" in str(failure.value)
    assert counted.calls == 3 * CORRELATION_SCALE_SAMPLES
    # The same model resolves a narrowed interval within budget.
    counted.calls = 0
    result = recover_rv(counted, observed, variance, search_min_m_s=-9, search_max_m_s=9)
    assert abs(result.rv_m_s - 3.3) < 1e-4
    assert result.model_evaluations == counted.calls
    # A budget below even the fixed work is refused before any model call.
    counted.calls = 0
    with pytest.raises(RVSearchBudgetError, match="raise max_evaluations"):
        recover_rv(counted, observed, variance, max_evaluations=1_000)
    assert counted.calls == 0
    with pytest.raises(ValueError, match="max_evaluations"):
        recover_rv(counted, observed, variance, max_evaluations=0)
