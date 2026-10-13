"""Synthetic, numerically motivated spectral/RV validation; no on-sky validation."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_distribution
from scipy.integrate import trapezoid

from aophoton.detector import DetectorConfig, simulate_detector
from aophoton.radial_velocity import recover_rv, rv_uncertainty
from aophoton.spectrograph import SpectrographConfig, SpectrographModel
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
    # Variances of Gaussian intrinsic lines and Gaussian LSFs add. Interpolation
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
