"""Scientific checks of diffraction, PSD normalization, and modal projection."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, j1

from aophoton.atmosphere import (
    ARCSEC_TO_RAD,
    make_ao_residual,
    remove_piston_tiptilt,
    von_karman_opd,
)
from aophoton.coupling import gaussian_mode, overlap_efficiency
from aophoton.lantern import ModeProjector, lantern_efficiency, modal_coupling, mode_basis
from aophoton.metrics import marechal_strehl, rms_wfe_nm, strehl_ratio
from aophoton.propagation import MAX_TILT_FIELD_FRACTION, RAD_TO_MAS, polychromatic_psf, propagate
from aophoton.pupil import make_pupil
from aophoton.config import load_config
from aophoton.simulation import simulate_optics

ROOT = Path(__file__).resolve().parents[1]


def test_unobscured_psf_matches_airy_profile():
    pupil = make_pupil(n_pix=128, obscuration=0)
    wavelength_um = 1.55
    focal = propagate(pupil, np.zeros_like(pupil.amplitude), wavelength_um, pad_factor=8)
    centre = len(focal.axis_mas) // 2
    theta_lam_over_d = focal.axis_mas / (wavelength_um * 1e-6 / pupil.diameter_m * RAD_TO_MAS)
    q = np.pi * theta_lam_over_d
    analytic = np.ones_like(q)
    nonzero = q != 0
    analytic[nonzero] = (2 * j1(q[nonzero]) / q[nonzero])**2
    measured = focal.intensity[centre] / focal.intensity[centre, centre]
    # Binary rasterization of a 128-pixel aperture perturbs the edge at O(1/N).
    # A 0.3% peak-intensity error over the first rings bounds that discretization.
    central_lobes = np.abs(theta_lam_over_d) < 3.5
    np.testing.assert_allclose(measured[central_lobes], analytic[central_lobes], atol=0.003, rtol=0)
    near_zero = (theta_lam_over_d > 0.9) & (theta_lam_over_d < 1.5)
    first_minimum = theta_lam_over_d[near_zero][np.argmin(measured[near_zero])]
    # The sampled first zero is within one 1/8 lambda/D angular pixel of 1.2197.
    assert abs(first_minimum - 1.21967) < 1 / 8


@pytest.mark.parametrize("n_pix,pad_factor", [(64, 1), (64, 4), (65, 3)])
def test_parseval_and_diffraction_limited_strehl(n_pix, pad_factor):
    pupil = make_pupil(n_pix=n_pix, obscuration=0.25, spider_width_m=0.06)
    zero = np.zeros_like(pupil.amplitude)
    reference = propagate(pupil, zero, 1.55, pad_factor)
    aberrated = propagate(pupil, make_ao_residual(pupil, 450, np.random.default_rng(8)), 1.55, pad_factor)
    # Unitary FFT energy should agree to floating-point roundoff, independently
    # of zero-padding, pupil obscuration, and phase error.
    assert reference.intensity.sum() == pytest.approx(1, abs=2e-14)
    assert aberrated.intensity.sum() == pytest.approx(1, abs=2e-14)
    assert np.sum(abs(aberrated.field)**2) == pytest.approx(1, abs=2e-14)
    assert strehl_ratio(reference.intensity, reference.intensity) == pytest.approx(1, abs=1e-14)
    peak = np.unravel_index(np.argmax(reference.intensity), reference.intensity.shape)
    assert peak == (n_pix * pad_factor // 2, n_pix * pad_factor // 2)


def test_absolute_von_karman_psd_normalization():
    pupil = make_pupil(n_pix=32)
    seeing, wavelength_um, outer_scale_m = 0.8, 0.5, 25.0
    rng = np.random.default_rng(234)
    measured_variances = np.array([
        np.mean(von_karman_opd(pupil, seeing, rng, wavelength_um, outer_scale_m)**2)
        for _ in range(96)
    ])
    frequency = np.fft.fftfreq(32, d=pupil.pixel_scale_m)
    fx, fy = np.meshgrid(frequency, frequency)
    lambda_m = wavelength_um * 1e-6
    r0_m = 0.98 * lambda_m / (seeing * ARCSEC_TO_RAD)
    psd = 0.023 * r0_m**(-5 / 3) * (fx**2 + fy**2 + outer_scale_m**-2)**(-11 / 6)
    psd[0, 0] = 0
    # The finite periodic screen has exactly these represented frequency bins;
    # do not compare to infinite-aperture variance with omitted subharmonics.
    variance_per_bin = psd / pupil.diameter_m**2 * (lambda_m / (2 * np.pi))**2
    expected_variance = variance_per_bin.sum()
    # For real Gaussian Fourier modes, variance of the averaged quadratic form
    # is 2 * sum(bin_variance²) / number_of_independent_screens.
    standard_error = np.sqrt(2 * np.sum(variance_per_bin**2) / len(measured_variances))
    assert abs(measured_variances.mean() - expected_variance) < 5 * standard_error


def test_residual_rms_seed_seeing_and_plane_removal():
    pupil = make_pupil(n_pix=64)
    first = make_ao_residual(pupil, 125, np.random.default_rng(17))
    repeated = make_ao_residual(pupil, 125, np.random.default_rng(17))
    worse_seeing = make_ao_residual(pupil, 125, np.random.default_rng(17), seeing_arcsec=1.2)
    np.testing.assert_array_equal(first, repeated)
    assert rms_wfe_nm(first, pupil) == pytest.approx(125, rel=2e-14)
    assert rms_wfe_nm(worse_seeing, pupil) == pytest.approx(125 * (1.2 / 0.8)**(5 / 6), rel=2e-14)
    for component in (np.ones_like(first), pupil.x_m, pupil.y_m):
        assert abs(np.sum(first * component * pupil.amplitude**2)) < 1e-17
    plane = 12e-9 + pupil.x_m * 16e-9 + pupil.y_m * 24e-9
    assert np.max(np.abs(remove_piston_tiptilt(plane, pupil))) < 1e-21


@pytest.mark.parametrize("wfe_nm", [15, 35, 60])
def test_marechal_small_aberration_limit(wfe_nm):
    pupil = make_pupil(n_pix=64)
    reference = propagate(pupil, np.zeros_like(pupil.amplitude), 1.55, 4)
    measured = []
    for seed in range(8):
        focal = propagate(pupil, make_ao_residual(pupil, wfe_nm, np.random.default_rng(seed)), 1.55, 4)
        measured.append(strehl_ratio(focal.intensity, reference.intensity))
    # At the largest sigma_phi=0.243 rad, fourth-order terms scale as 0.0035.
    # This 0.4% absolute bound allows finite-screen non-Gaussian phase moments.
    np.testing.assert_allclose(measured, marechal_strehl(wfe_nm, 1.55), atol=0.004, rtol=0)


def test_tip_tilt_sign_and_single_mode_loss():
    pupil = make_pupil(n_pix=64)
    zero = np.zeros_like(pupil.amplitude)
    reference = propagate(pupil, zero, 1.55, 4)
    mode = gaussian_mode(reference, 57)
    displacement = 3 * reference.pixel_scale_mas
    shifted = propagate(pupil, zero, 1.55, 4, (displacement, 0))
    centre = len(reference.axis_mas) // 2
    assert np.unravel_index(np.argmax(shifted.intensity), shifted.intensity.shape) == (centre, centre + 3)
    assert overlap_efficiency(shifted.field, mode) < overlap_efficiency(reference.field, mode)
    # An exact integer-pixel static displacement preserves the sampled peak.
    assert strehl_ratio(shifted.intensity, reference.intensity) == pytest.approx(1, abs=2e-14)


def test_tilt_that_would_wrap_round_the_fft_field_is_refused():
    # 32 px at 1.55 um: the periodic field spans +/-1421 mas. A +1415 mas tilt
    # used to pass and put the peak at -1421 mas, with 55% of the power at x < 0.
    small, large = make_pupil(n_pix=32), make_pupil(n_pix=128)
    with pytest.raises(ValueError, match="wrap round the FFT field. Increase telescope.n_pix"):
        propagate(small, np.zeros_like(small.amplitude), 1.55, 4, (1415.0, 0.0))
    with pytest.raises(ValueError, match="per axis"):
        propagate(small, np.zeros_like(small.amplitude), 1.55, 4, (0.0, -1415.0))
    half_field = 1.55e-6 / (2 * small.pixel_scale_m) * RAD_TO_MAS
    tilt = MAX_TILT_FIELD_FRACTION * half_field          # largest accepted tilt
    # A 4x wider field with the same angular pixel scale shows the unwrapped image.
    wrapped = propagate(small, np.zeros_like(small.amplitude), 1.55, 4, (tilt, 0.0))
    reference = propagate(large, np.zeros_like(large.amplitude), 1.55, 4, (tilt, 0.0))
    assert wrapped.pixel_scale_mas == pytest.approx(reference.pixel_scale_mas, rel=1e-12)

    def signed_peak_and_far_power(focal):
        row, column = np.unravel_index(np.argmax(focal.intensity), focal.intensity.shape)
        return focal.axis_mas[column], focal.intensity[:, focal.axis_mas < 0].sum()

    peak, far = signed_peak_and_far_power(wrapped)
    reference_peak, reference_far = signed_peak_and_far_power(reference)
    assert peak == pytest.approx(reference_peak, abs=1e-9) and peak > 0
    # Wrapped wings add ~0.5% of the power to the far half-plane at the limit
    # (1.08% -> 1.61%); the coarser 32 px pupil itself accounts for ~0.1%.
    assert abs(far - reference_far) < 0.01
    # Pointing well inside the limit is propagated exactly as before.
    inside = propagate(small, np.zeros_like(small.amplitude), 1.55, 4, (0.3 * half_field, -0.2 * half_field))
    assert inside.intensity.sum() == pytest.approx(1, abs=2e-14)


def test_tilt_carried_by_the_opd_is_guarded_like_explicit_pointing():
    # The same +1415.38 mas x tilt as an OPD ramp, as pointing, or split between
    # them over a residual aberration.  At 32 px (limit 710.47 mas) the OPD
    # route used to pass and put the peak at -1420.94 mas with 55% of the power
    # at x < 0, while equal and opposite contributions were refused.
    tilt, wavelength_um, pad = 1415.38, 1.55, 4
    small, large = make_pupil(n_pix=32, diameter_m=3.6), make_pupil(n_pix=128, diameter_m=3.6)

    def ramp(pupil, mas):
        return pupil.x_m * mas / RAD_TO_MAS

    zero = np.zeros_like(small.amplitude)
    residual = make_ao_residual(small, 150, np.random.default_rng(5))       # tip-tilt removed
    for opd, pointing in ((ramp(small, tilt), (0.0, 0.0)), (zero, (tilt, 0.0)),
                          (0.6 * ramp(small, tilt) + residual, (0.4 * tilt, 0.0))):
        with pytest.raises(ValueError, match="wrap round the FFT field. Increase telescope.n_pix"):
            propagate(small, opd, wavelength_um, pad, pointing)
    flat = propagate(small, zero, wavelength_um, pad)
    cancelled = propagate(small, -ramp(small, tilt), wavelength_um, pad, (tilt, 0.0))
    np.testing.assert_allclose(cancelled.intensity, flat.intensity, rtol=0, atol=1e-12 * flat.intensity.max())
    # 128 px at the same angular pixel scale: both encodings give one PSF, peaked
    # on the +x side at the grid sample nearest the tilt (+1420.94 mas).
    zero = np.zeros_like(large.amplitude)
    by_opd = propagate(large, ramp(large, tilt), wavelength_um, pad)
    by_pointing = propagate(large, zero, wavelength_um, pad, (tilt, 0.0))
    assert by_opd.pixel_scale_mas == pytest.approx(flat.pixel_scale_mas, rel=1e-12)
    for focal in (by_opd, by_pointing):
        np.testing.assert_allclose(focal.intensity, by_pointing.intensity, rtol=0, atol=1e-12 * focal.intensity.max())
        row, column = np.unravel_index(np.argmax(focal.intensity), focal.intensity.shape)
        assert focal.axis_mas[column] == pytest.approx(1420.94, abs=0.01) and row == focal.axis_mas.size // 2
        assert focal.intensity[:, focal.axis_mas < 0].sum() < 0.01
    # A split tilt keeps the residual aberration it is carried with.
    residual = make_ao_residual(large, 150, np.random.default_rng(5))
    split = propagate(large, 0.6 * ramp(large, tilt) + residual, wavelength_um, pad, (0.4 * tilt, 0.0))
    whole = propagate(large, ramp(large, tilt) + residual, wavelength_um, pad)
    np.testing.assert_allclose(split.intensity, whole.intensity, rtol=0, atol=1e-12 * whole.intensity.max())
    assert strehl_ratio(split.intensity, by_opd.intensity) < 0.95


def test_normalized_complex_overlap_and_ensemble_degradation():
    pupil = make_pupil(n_pix=64)
    reference = propagate(pupil, np.zeros_like(pupil.amplitude), 1.55, 4)
    mode = gaussian_mode(reference, 57)
    assert overlap_efficiency(mode, mode * (2 + 4j)) == pytest.approx(1, abs=2e-14)
    # Equal intensities with opposing phases can have exactly zero coupling.
    assert overlap_efficiency([[1, -1], [1, -1]], np.ones((2, 2))) == 0
    ideal = overlap_efficiency(reference.field, mode)
    moderate, severe = [], []
    for seed in range(16):
        opd = make_ao_residual(pupil, 150, np.random.default_rng(seed))
        for residual, results in ((opd, moderate), (opd * (600 / 150), severe)):
            efficiency = overlap_efficiency(propagate(pupil, residual, 1.55, 4).field, mode)
            assert 0 <= efficiency <= 1
            results.append(efficiency)
    difference = np.asarray(moderate) - severe
    # Paired independent atmospheric realizations: the large-aberration loss
    # must exceed three standard errors, not just decrease in one lucky draw.
    assert ideal > np.mean(moderate)
    assert difference.mean() > 3 * difference.std(ddof=1) / np.sqrt(len(difference))


def test_lantern_orthogonality_monotonicity_and_gaussian_fundamental():
    pupil = make_pupil(n_pix=48)
    focal = propagate(pupil, make_ao_residual(pupil, 250, np.random.default_rng(2)), 1.55, 4)
    basis = mode_basis(focal, 57, 10)
    flattened = basis.reshape(10, -1)
    np.testing.assert_allclose(flattened @ flattened.T, np.eye(10), atol=2e-14, rtol=0)
    np.testing.assert_allclose(basis[0], gaussian_mode(focal, 57), atol=2e-15, rtol=0)
    powers = modal_coupling(focal.field, basis)
    np.testing.assert_allclose(powers[0], overlap_efficiency(focal.field, basis[0]), atol=2e-14)
    assert 0 <= powers.sum() <= 1
    efficiencies = [lantern_efficiency(focal.field, basis, ports, transmission=0.9) for ports in (1, 3, 6, 10)]
    assert np.all(np.diff(efficiencies) >= -1e-14)
    assert efficiencies[-1] == pytest.approx(0.9 * powers.sum(), abs=2e-14)
    np.testing.assert_allclose(mode_basis(focal, 57, 3), basis[:3], atol=2e-14, rtol=0)
    # Duplicate modes would double-count power and must not be silently used.
    with pytest.raises(ValueError, match="orthonormal"):
        modal_coupling(focal.field, np.stack([basis[0], basis[0]]))


def test_reusable_projector_matches_projection_and_owns_its_basis():
    pupil = make_pupil(n_pix=32)
    focal = propagate(pupil, make_ao_residual(pupil, 220, np.random.default_rng(22)), 1.55, 4)
    basis = mode_basis(focal, 57, 6)
    expected = modal_coupling(focal.field, basis)
    projector = ModeProjector(basis)
    basis[:] = 0
    np.testing.assert_allclose(projector.powers(focal.field), expected, atol=2e-14, rtol=0)
    with pytest.raises(ValueError, match="match"):
        projector.powers(np.ones((2, 2)))


@pytest.mark.parametrize("basis", [
    np.array([[[True, True], [False, False]]]),                 # norm^2 = 2, saturated to True in bool
    np.array([[[16, 1], [0, 0]]], dtype=np.int8),               # norm^2 = 257, wraps to 1 in int8
], ids=["bool", "int8"])
def test_non_normalised_basis_fails_in_any_sample_dtype(basis):
    field = np.eye(2)
    with pytest.raises(ValueError, match="orthonormal"):
        modal_coupling(field, basis)
    with pytest.raises(ValueError, match="orthonormal"):
        ModeProjector(basis)
    with pytest.raises(ValueError, match="numbers"):
        modal_coupling(field, np.array([[["a", "b"], ["c", "d"]]]))


def test_normalised_real_and_complex_bases_keep_their_powers():
    field = np.array([[1.0, 2.0j], [0.0, -1.0]])
    power = np.vdot(field, field).real
    delta = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])         # integer pixel modes
    expected = np.abs(field.ravel()[:2])**2 / power
    for basis in (delta, delta.astype(bool), delta.astype(float)):
        np.testing.assert_allclose(modal_coupling(field, basis), expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(ModeProjector(basis).powers(field), expected, rtol=0, atol=1e-15)
    # A complex, phase-rotated orthonormal pair spanning the first row.
    rotated = np.array([[[1, 1j], [0, 0]], [[1, -1j], [0, 0]]]) * np.exp(0.7j) / np.sqrt(2)
    rotated_expected = np.abs(rotated.reshape(2, -1).conj() @ field.ravel())**2 / power
    np.testing.assert_allclose(modal_coupling(field, rotated), rotated_expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(ModeProjector(rotated).powers(field), rotated_expected, rtol=0, atol=1e-15)
    assert modal_coupling(field, rotated).sum() == pytest.approx(expected.sum(), abs=1e-15)


def test_polychromatic_flux_and_explicit_crop_loss():
    pupil = make_pupil(n_pix=48)
    zero = np.zeros_like(pupil.amplitude)
    mono = propagate(pupil, zero, 1.55, 4)
    duplicate = polychromatic_psf(pupil, zero, [1.55, 1.55], weights=[1, 3])
    np.testing.assert_allclose(duplicate.intensity, mono.intensity, atol=2e-14, rtol=0)
    broad = polychromatic_psf(pupil, zero, [1.50, 1.55, 1.60])
    assert broad.intensity.sum() == pytest.approx(1, abs=3e-13)
    centre = len(mono.axis_mas) // 2
    cropped = polychromatic_psf(pupil, zero, [1.50, 1.55, 1.60], axis_mas=mono.axis_mas[centre - 2:centre + 3])
    assert 0 < cropped.intensity.sum() < 0.8
    assert np.all(cropped.intensity >= 0)


def _vane_integral(width_m, inner_radius_m, outer_radius_m, profile):
    """Integral of profile(x) * g(x) over the vertical vane |x| < w/2 inside the
    annulus, where g(x) is the y-integral that ``profile`` returns as a pair
    (x-weight, callable of the y-limits)."""
    def integrand(x):
        high = np.sqrt(outer_radius_m**2 - x * x)
        low = np.sqrt(max(inner_radius_m**2 - x * x, 0.0))
        weight, along = profile(x)
        return weight * 2 * (along(high) - along(low))
    return quad(integrand, -width_m / 2, width_m / 2, epsabs=0, epsrel=1e-12)[0]


def _continuous_vane_coupling(width_m, wavelength_um=1.555, mode_radius_mas=57.0,
                              outer_radius_m=1.8, inner_radius_m=0.45):
    """Flat-wavefront SMF coupling of an annulus with crossed vanes, computed in
    the pupil plane without sampling: eta = |int_P u|^2 / (A_P int |u|^2),
    with the fibre mode back-propagated to u = exp(-a r^2). Valid while the
    vanes miss the central overlap, i.e. w < sqrt(2) * inner radius."""
    a = (np.pi * (mode_radius_mas / RAD_TO_MAS) / (wavelength_um * 1e-6))**2
    gaussian = lambda x: (np.exp(-a * x * x), lambda y: 0.5 * np.sqrt(np.pi / a) * erf(np.sqrt(a) * y))
    flat = lambda x: (1.0, lambda y: y)
    blocked = 2 * _vane_integral(width_m, inner_radius_m, outer_radius_m, gaussian)
    blocked_area = 2 * _vane_integral(width_m, inner_radius_m, outer_radius_m, flat)
    field = np.pi / a * (np.exp(-a * inner_radius_m**2) - np.exp(-a * outer_radius_m**2)) - blocked
    area = np.pi * (outer_radius_m**2 - inner_radius_m**2) - blocked_area
    return field**2 / (area * np.pi / (2 * a)), area


@pytest.mark.parametrize("width_m", [0.001, 0.3])
def test_spider_removes_its_geometric_area(width_m):
    # A 1 mm crossed spider on a clear 3.6 m pupil blocks 0.007199 m^2. Masking
    # every pixel whose centre lay within the vane removed 0.772 m^2 at 32 px.
    exact = 8 * quad(lambda x: np.sqrt(1.8**2 - x * x), 0, width_m / 2)[0] - width_m**2
    errors = []
    for n_pix in (32, 64, 128):
        clear, vaned = make_pupil(n_pix, 3.6, 0), make_pupil(n_pix, 3.6, 0, width_m)
        removed = clear.clear_area_m2 - vaned.clear_area_m2
        errors.append(abs(removed / exact - 1))
        # Only the binary outer edge is approximated: the vane's chord is
        # sampled to one pixel, a relative error up to pixel / diameter.
        assert errors[-1] < 1.05 / n_pix
    assert errors[0] > errors[1] > errors[2]
    pupil = make_pupil(32, 3.6, 0, 0.001)
    centre = 16
    assert pupil.amplitude[5, centre] == pytest.approx(1 - 0.001 / pupil.pixel_scale_m, rel=1e-12)
    assert pupil.amplitude[centre, centre] == pytest.approx((1 - 0.001 / pupil.pixel_scale_m)**2, rel=1e-12)
    assert make_pupil(32, 3.6, 0).sampled_power_fraction == 1.0


@pytest.mark.parametrize("width_m", [0.001, 0.01, 0.06])
def test_spider_coupling_change_matches_continuous_pupil_overlap(width_m):
    expected = _continuous_vane_coupling(width_m)[0] / _continuous_vane_coupling(0.0)[0] - 1
    changes = []
    for n_pix in (64, 128):
        clear, vaned = make_pupil(n_pix, 3.6, 0.25), make_pupil(n_pix, 3.6, 0.25, width_m)
        overlaps = []
        for pupil in (clear, vaned):
            focal = propagate(pupil, np.zeros_like(pupil.amplitude), 1.555, 4)
            overlaps.append(overlap_efficiency(focal.field, gaussian_mode(focal, 57.0)))
        changes.append(overlaps[1] * vaned.sampled_power_fraction / overlaps[0] - 1)
        # The diffraction response converges at O(pixel / diameter), like the area.
        assert changes[-1] == pytest.approx(expected, rel=3.0 / n_pix)
        if width_m < vaned.pixel_scale_m:
            # A sub-pixel vane is all partial coverage: without referring the
            # sampled overlap to the collected photons, over half its
            # coupling loss would be missed.
            assert abs(overlaps[1] / overlaps[0] - 1 - expected) > 0.4 * abs(expected)
    assert abs(changes[1] - expected) < abs(changes[0] - expected)


def test_simulation_collects_over_the_clear_area():
    config = load_config(ROOT / "configs/baseline.yaml")
    config = replace(config, telescope=replace(config.telescope, n_pix=32, spider_width_m=0.03),
                     ao=replace(config.ao, n_frames=1, residual_wfe_nm=0.0, tip_tilt_rms_mas=0.0, wavelength_nodes=2))
    optics = simulate_optics(config)
    pupil = make_pupil(32, config.telescope.diameter_m, config.telescope.obscuration, 0.03)
    assert optics.collecting_area_m2 == pytest.approx(pupil.clear_area_m2, rel=1e-15)
    focal = propagate(pupil, np.zeros_like(pupil.amplitude), optics.wavelengths_um[0], config.telescope.pad_factor)
    raw = overlap_efficiency(focal.field, gaussian_mode(focal, config.fibre.mode_radius_mas))
    assert optics.smf_efficiency[0] == pytest.approx(raw * pupil.sampled_power_fraction, rel=1e-12)
    # Coupled photons per unit flux: geometric area x photon-referred overlap
    # equals the sampled-field power x the sampled overlap.
    sampled_power = np.sum(pupil.amplitude**2) * pupil.pixel_scale_m**2
    assert optics.collecting_area_m2 * optics.smf_efficiency[0] == pytest.approx(sampled_power * raw, rel=1e-12)


@pytest.mark.parametrize("kwargs", [{"n_pix": 7}, {"obscuration": 1}, {"diameter_m": 0}, {"spider_width_m": -1}])
def test_invalid_pupil_parameters(kwargs):
    with pytest.raises(ValueError):
        make_pupil(**kwargs)


def test_unresolved_or_truncated_fibre_mode_is_rejected():
    # A 1 mas mode on the default ~22 mas grid would be normalised into a
    # discrete delta whose coupling scales with the pixel size (0.046 at pad 4,
    # ~79x the continuous overlap); a mode wider than the grid is truncated.
    from aophoton.coupling import gaussian_mode
    from aophoton.lantern import mode_basis
    from aophoton.propagation import propagate
    from aophoton.pupil import make_pupil

    pupil = make_pupil(128, 3.6, 0.25, 0.0)
    focal = propagate(pupil, np.zeros_like(pupil.amplitude), 1.555, 4)
    with pytest.raises(ValueError, match="pad_factor"):
        gaussian_mode(focal, 1.0)
    with pytest.raises(ValueError, match="pad_factor"):
        mode_basis(focal, 1.0, 6)
    with pytest.raises(ValueError, match="truncated"):
        gaussian_mode(focal, 3000.0)
    # The default 57 mas mode is resolved (~2.6 px) and converged under padding.
    from aophoton.coupling import overlap_efficiency
    fine = propagate(pupil, np.zeros_like(pupil.amplitude), 1.555, 16)
    assert overlap_efficiency(focal.field, gaussian_mode(focal, 57.0)) == pytest.approx(
        overlap_efficiency(fine.field, gaussian_mode(fine, 57.0)), rel=1e-3)


def test_higher_order_modes_require_finer_sampling():
    # A complete degree-8 shell (45 modes, 29 mas) at padding 4 has 1.30 px per
    # radius: accepted before, it overestimated displaced-field acceptance by 62%.
    # It is now rejected; padding 8 (2.60 px) is accepted and converged.
    from aophoton.coupling import required_pixels_per_radius
    from aophoton.lantern import modal_coupling, mode_basis
    from aophoton.propagation import propagate
    from aophoton.pupil import make_pupil

    pupil = make_pupil(32)
    zero = np.zeros_like(pupil.amplitude)
    with pytest.raises(ValueError, match="degree 8"):
        mode_basis(propagate(pupil, zero, 1.555, 4, (150, 0)), 29, 45)
    accepted = [float(modal_coupling(focal.field, mode_basis(focal, 29, 45)).sum())
                for focal in (propagate(pupil, zero, 1.555, pad, (150, 0)) for pad in (8, 16))]
    assert accepted[0] == pytest.approx(accepted[1], rel=1e-4)
    assert required_pixels_per_radius(0) < required_pixels_per_radius(2) < required_pixels_per_radius(8)


@pytest.mark.parametrize("degree", [0, 2, 4, 8])
def test_minimum_accepted_sampling_is_converged_for_displaced_fields(degree):
    # At the smallest padding the guard accepts, modal acceptance of displaced
    # and aberrated fields agrees with a padding-16 reference to < 0.1%.
    from aophoton.coupling import required_pixels_per_radius
    from aophoton.lantern import modal_coupling, mode_basis
    from aophoton.propagation import propagate
    from aophoton.pupil import make_pupil

    pupil = make_pupil(32)
    n_modes = (degree + 1) * (degree + 2) // 2
    pixel_pad2 = propagate(pupil, np.zeros_like(pupil.amplitude), 1.555, 2).pixel_scale_mas
    radius = required_pixels_per_radius(degree) * pixel_pad2 * 1.0001       # just accepted at padding 2
    rng = np.random.default_rng(degree)
    for opd in (np.zeros_like(pupil.amplitude), make_ao_residual(pupil, 300, rng)):
        for shift in (0.0, 2.0 * radius):
            coarse, fine = (propagate(pupil, opd, 1.555, pad, (shift, 0.0)) for pad in (2, 16))
            a = float(modal_coupling(coarse.field, mode_basis(coarse, radius, n_modes)).sum())
            b = float(modal_coupling(fine.field, mode_basis(fine, radius, n_modes)).sum())
            assert abs(a - b) <= 1e-3 * max(b, 1e-3)


def test_sampling_message_does_not_claim_calibration_beyond_its_range():
    from aophoton.coupling import CALIBRATED_MAX_DEGREE
    from aophoton.lantern import mode_basis
    from aophoton.propagation import propagate
    from aophoton.pupil import make_pupil

    focal = propagate(make_pupil(32), np.zeros((32, 32)), 1.555, 2)
    with pytest.raises(ValueError, match=r"<0\.1%"):
        mode_basis(focal, 29, 45)                                   # degree 8: calibrated
    with pytest.raises(ValueError, match="extrapolated beyond degree 8"):
        mode_basis(focal, 29, 66)                                   # degree 10
    assert CALIBRATED_MAX_DEGREE == 8
