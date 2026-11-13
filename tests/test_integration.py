"""Public configuration, deterministic end-to-end chain, and CLI output contract."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pytest

from aophoton.cli import main
from aophoton.config import config_from_dict, load_config, parameter_records
from aophoton.simulation import (
    SYNTHETIC_TEMPLATE_MARGIN_UM, run_simulation, save_result, simulation_template, synthetic_template_range_um,
)
from aophoton.spectrograph import template_support_um
from aophoton.spectrum import synthetic_spectrum

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def baseline():
    return run_simulation(load_config(ROOT / "configs/baseline.yaml"))


def test_baseline_end_to_end_and_seed(baseline):
    repeat = run_simulation(baseline.config)
    np.testing.assert_array_equal(baseline.optics.mean_psf, repeat.optics.mean_psf)
    np.testing.assert_array_equal(baseline.coupling, repeat.coupling)
    np.testing.assert_array_equal(baseline.detector.flux_e, repeat.detector.flux_e)
    assert baseline.rv.rv_m_s == repeat.rv.rv_m_s
    assert 0 < baseline.optics.strehl < 1
    assert np.all((baseline.coupling >= 0) & (baseline.coupling <= 1))
    # A single fixed-seed draw should lie inside a broad 4-sigma acceptance
    # interval; the separate 64-draw test checks calibration statistically.
    assert abs(baseline.rv.rv_m_s-baseline.config.observation.injected_rv_m_s) < 4*baseline.rv.uncertainty_m_s
    assert not baseline.rv.at_boundary


def test_exported_outputs_retain_all_provenance(baseline, tmp_path):
    save_result(baseline, tmp_path)
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["configuration"] == asdict(baseline.config)
    records = payload["parameter_provenance"]
    flattened_count = sum(len(v) if isinstance(v, dict) else 1 for v in asdict(baseline.config).values())
    assert len(records) == flattened_count
    assert sum(r["status"] == "sourced" for r in records) == 3
    data = np.genfromtxt(tmp_path / "spectrum.csv", delimiter=",", names=True)
    np.testing.assert_allclose(data["expected_source_e"], baseline.detector.expected_source_e)
    with np.load(tmp_path / "arrays.npz") as arrays:
        assert arrays["psf"].sum() == pytest.approx(1, abs=3e-14)


def test_cli_noiseless_lantern(tmp_path, capsys):
    import yaml
    config = load_config(ROOT / "configs/baseline.yaml")
    # Smaller ensemble exercises the same CLI/operator without duplicating
    # the full baseline's optical statistics in this command-line test.
    # Padding 3 keeps the six-mode basis converged (degree 2 needs >= 1.79 px per radius).
    config = replace(config, telescope=replace(config.telescope,n_pix=48,pad_factor=3),
                     ao=replace(config.ao,n_frames=2),
                     fibre=replace(config.fibre,feed="lantern",lantern_ports=3))
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    out = tmp_path / "out"
    assert main(["simulate","--config",str(path),"--output",str(out),"--no-noise"]) == 0
    payload = json.loads((out / "summary.json").read_text())
    # 0.01 m/s is 1e-5 detector pixels; this is a same-operator numerical check.
    assert abs(payload["recovery_bias_m_s"]) < 0.01
    assert payload["feed"] == "lantern"
    assert "Outputs:" in capsys.readouterr().out


def test_built_in_template_covers_the_model_at_low_resolving_power(tmp_path):
    config = load_config(ROOT / "configs/baseline.yaml")
    s = config.spectrograph
    # Default: the fixed 5 nm margin already covers the model, so the seeded
    # spectrum is byte-for-byte the one the published results used.
    legacy = (s.wavelength_min_um - SYNTHETIC_TEMPLATE_MARGIN_UM, s.wavelength_max_um + SYNTHETIC_TEMPLATE_MARGIN_UM)
    assert synthetic_template_range_um(s) == legacy
    np.testing.assert_array_equal(simulation_template(config).relative_flux,
                                  synthetic_spectrum(*legacy, seed=config.observation.template_seed).relative_flux)
    # At R = 500 the six-sigma LSF alone reaches ~8 nm beyond the band; the
    # fixed margin made the normal simulation fail its coverage check.
    small = replace(config, telescope=replace(config.telescope, n_pix=32), ao=replace(config.ao, n_frames=1),
                    spectrograph=replace(s, resolving_power=500.0))
    low, high = synthetic_template_range_um(small.spectrograph)
    support = template_support_um(small.spectrograph)
    assert low <= support[0] < legacy[0] and high >= support[1] > legacy[1]
    result = run_simulation(small, noisy=False)
    assert abs(result.rv.rv_m_s - small.observation.injected_rv_m_s) < 0.01
    # An imported template is still checked, never silently extended.
    path = tmp_path / "legacy_range.txt"
    template = synthetic_spectrum(*legacy, seed=7)
    np.savetxt(path, np.column_stack([template.wavelength_um, template.relative_flux]))
    imported = replace(small, observation=replace(small.observation, template_path=str(path)))
    with pytest.raises(ValueError, match="Template must cover"):
        run_simulation(imported, noisy=False)


def test_flat_file_template_is_reported_as_carrying_no_rv(tmp_path):
    # Through the public simulator: a file template at 0.5 over every reachable
    # wavelength (1.0 only beyond) gave 2471.45 m/s with the boundary flag set.
    wavelength = np.geomspace(1.53, 1.58, 20_000)
    template = tmp_path / "plateau.csv"
    np.savetxt(template, np.column_stack((wavelength, np.where((wavelength > 1.538) & (wavelength < 1.572), 0.5, 1.0))),
               delimiter=",")
    config = load_config(ROOT / "configs/baseline.yaml")
    config = replace(config, telescope=replace(config.telescope, n_pix=32), ao=replace(config.ao, n_frames=2),
                     observation=replace(config.observation, template_path=str(template)))
    with pytest.raises(ValueError, match="no RV information"):
        run_simulation(config, noisy=False)


def _small_cli_config(tmp_path, **observation):
    import yaml
    config = load_config(ROOT / "configs/baseline.yaml")
    config = replace(config, telescope=replace(config.telescope, n_pix=48, pad_factor=3),
                     ao=replace(config.ao, n_frames=2), observation=replace(config.observation, **observation))
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    return path


def test_noise_mode_and_template_identity_are_recorded_and_replayable(tmp_path, capsys):
    template = synthetic_spectrum(1.535, 1.575, seed=11)
    template_path = tmp_path / "template.txt"
    np.savetxt(template_path, np.column_stack([template.wavelength_um, template.relative_flux]))
    config = _small_cli_config(tmp_path, template_path=str(template_path))
    runs = {}
    for mode, flags in (("noisy", []), ("noiseless", ["--no-noise"])):
        out = tmp_path / mode
        assert main(["simulate", "--config", str(config), "--output", str(out), *flags]) == 0
        payload = json.loads((out / "summary.json").read_text())
        runs[mode] = payload
        assert payload["noise_mode"] == mode
        assert payload["template"]["source"] == "file"
        assert payload["template"]["file_sha256"] == hashlib.sha256(template_path.read_bytes()).hexdigest()
        assert payload["software_versions"]["python"] == platform.python_version()
        # The CSV header is the documented column list; the observed column says what it holds.
        columns = list(payload["spectrum_csv_columns"])
        header = (out / "spectrum.csv").read_text().splitlines()[0].split(",")
        assert header == columns
        observed = "background_subtracted_noisy_e" if mode == "noisy" else "background_subtracted_expected_e"
        assert columns[2] == observed and "snr_per_spectral_sample" in columns
        data = np.genfromtxt(out / "spectrum.csv", delimiter=",", names=True)
        if mode == "noiseless":
            np.testing.assert_array_equal(data[observed], data["expected_source_e"])
        else:
            assert np.any(data[observed] != data["expected_source_e"])
        assert payload["detector_pixels_per_spectral_sample"] == 4
        assert "4 physical pixels" in payload["snr_definition"]
        assert payload["median_snr_per_pixel"] == payload["median_snr_per_spectral_sample"]
        # Each mode replays from its own summary alone.
        replay = tmp_path / f"{mode}_replay"
        assert main(["replay", "--summary", str(out / "summary.json"), "--output", str(replay)]) == 0
        assert "reproduces all" in capsys.readouterr().out
        replayed = json.loads((replay / "summary.json").read_text())
        assert replayed["run_identity_sha256"] == payload["run_identity_sha256"]
        assert replayed["noise_mode"] == mode
        assert (replay / "spectrum.csv").read_bytes() == (out / "spectrum.csv").read_bytes()
    # Same configuration and seed: the metadata, not just the numbers, tell the runs apart.
    assert runs["noisy"]["configuration"] == runs["noiseless"]["configuration"]
    assert runs["noisy"]["run_identity_sha256"] != runs["noiseless"]["run_identity_sha256"]
    # Changed template bytes change the identity, and the old summary refuses to replay.
    flux = template.relative_flux.copy()
    flux[1000] *= 0.99
    np.savetxt(template_path, np.column_stack([template.wavelength_um, flux]))
    with pytest.raises(SystemExit) as refused:
        main(["replay", "--summary", str(tmp_path / "noisy/summary.json"), "--output", str(tmp_path / "stale")])
    assert refused.value.code == 2
    assert "no longer matches the recorded SHA-256" in capsys.readouterr().err
    assert main(["simulate", "--config", str(config), "--output", str(tmp_path / "changed")]) == 0
    changed = json.loads((tmp_path / "changed/summary.json").read_text())
    assert changed["template"]["file_sha256"] != runs["noisy"]["template"]["file_sha256"]
    assert changed["run_identity_sha256"] != runs["noisy"]["run_identity_sha256"]


def _tree_bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


@pytest.fixture(scope="module")
def replay_reference(tmp_path_factory):
    """A noiseless smf-feed CLI run: the selected feed is not the lantern."""
    root = tmp_path_factory.mktemp("replay")
    config = _small_cli_config(root)
    assert main(["simulate", "--config", str(config), "--output", str(root / "pristine"), "--no-noise"]) == 0
    return root / "pristine"


def _edited_reference(source, target, **changes):
    """A copy of a recorded run whose summary differs only in ``changes``."""
    target.mkdir()
    for name, data in _tree_bytes(source).items():
        (target / name).write_bytes(data)
    summary = json.loads((source / "summary.json").read_text())
    assert all(key in summary for key in changes)
    summary.update(changes)
    (target / "summary.json").write_text(json.dumps(summary, indent=2))
    return target


def test_failed_replay_writes_nothing_even_into_its_own_directory(replay_reference, tmp_path, capsys):
    recorded = json.loads((replay_reference / "summary.json").read_text())
    assert recorded["feed"] == "smf" and abs(recorded["recovered_rv_m_s"] - 2300) < 1e-3
    reference = _edited_reference(replay_reference, tmp_path / "reference", recovered_rv_m_s=2301.0)
    alias = tmp_path / "alias"
    alias.symlink_to(reference, target_is_directory=True)
    populated = tmp_path / "populated"
    populated.mkdir()
    for name in ("summary.json", "spectrum.csv", "arrays.npz", "notes.txt"):
        (populated / name).write_bytes(f"earlier {name}".encode())
    before = {path: _tree_bytes(path) for path in (reference, populated)}
    summary = reference / "summary.json"
    # In place (twice: the first attempt must not "repair" the reference), through
    # a symlink and a dot-dot alias of the reference, and into a populated directory.
    for output in (reference, reference, alias, tmp_path / "alias/../reference", populated):
        assert main(["replay", "--summary", str(summary), "--output", str(output)]) == 1
        out = capsys.readouterr().out
        assert "differs in 1 of" in out and "recovered_rv_m_s: recorded 2301.0" in out
        assert "Nothing was written" in out
        assert {path: _tree_bytes(path) for path in before} == before
    # Unchanged, the same summary replays to a fresh destination ...
    pristine = replay_reference / "summary.json"
    fresh = tmp_path / "fresh"
    assert main(["replay", "--summary", str(pristine), "--output", str(fresh)]) == 0
    assert "reproduces all" in capsys.readouterr().out
    assert set(_tree_bytes(fresh)) == {"summary.json", "spectrum.csv", "arrays.npz"}
    assert (fresh / "spectrum.csv").read_bytes() == (replay_reference / "spectrum.csv").read_bytes()
    # ... and into its own directory, which it leaves exactly as recorded.
    unchanged = _tree_bytes(replay_reference)
    assert main(["replay", "--summary", str(pristine), "--output", str(replay_reference)]) == 0
    assert "reference directory, left unchanged" in capsys.readouterr().out
    assert _tree_bytes(replay_reference) == unchanged


@pytest.mark.parametrize(("field", "value"), [("mean_lantern_coupling", 0.99), ("rv_at_search_boundary", True),
                                               ("recovery_bias_m_s", 1.0), ("detector_pixels_per_spectral_sample", 5)])
def test_replay_names_any_changed_scientific_result(replay_reference, tmp_path, capsys, field, value):
    recorded = json.loads((replay_reference / "summary.json").read_text())
    assert recorded[field] != value
    reference = _edited_reference(replay_reference, tmp_path / "reference", **{field: value})
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "summary.json").write_text("earlier")
    before = {path: _tree_bytes(path) for path in (reference, destination)}
    assert main(["replay", "--summary", str(reference / "summary.json"), "--output", str(destination)]) == 1
    out = capsys.readouterr().out
    assert "differs in 1 of" in out and f"  {field}: recorded {value!r}, replayed {recorded[field]!r}" in out
    assert {path: _tree_bytes(path) for path in before} == before


def test_replay_refuses_a_summary_without_every_compared_result(replay_reference, tmp_path, capsys):
    reference = _edited_reference(replay_reference, tmp_path / "reference")
    summary = json.loads((reference / "summary.json").read_text())
    del summary["mean_few_mode_coupling"]
    (reference / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit) as refused:
        main(["replay", "--summary", str(reference / "summary.json"), "--output", str(tmp_path / "out")])
    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert "older summary schema" in error and "'mean_few_mode_coupling'" in error
    assert not (tmp_path / "out").exists()


def test_config_round_trips_through_its_summary_strictly(tmp_path):
    config = load_config(ROOT / "configs/baseline.yaml")
    assert config_from_dict(json.loads(json.dumps(asdict(config)))) == config
    raw = asdict(config)
    raw["ao"]["residul_wfe_nm"] = 100
    with pytest.raises(ValueError, match="Unknown keys"):
        config_from_dict(raw)


@pytest.mark.parametrize("text", ["ao: {residul_wfe_nm: 100}", "unknown: 1", "[]",
                                     "ao: {n_frames: 0}", "telescope: {obscuration: 1}",
                                     "fibre: {lantern_transmission: 1.1}", "seed: true",
                                     # +/-10 km/s search leaves no room for the 1 m/s Fisher step
                                     "spectrograph: {rv_padding_m_s: 10000.5}"])
def test_invalid_configuration_rejected(text, tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_modified_source_value_is_relabelled():
    config = load_config(ROOT / "configs/baseline.yaml")
    config = replace(config, telescope=replace(config.telescope, diameter_m=8))
    record = next(r for r in parameter_records(config) if r["parameter"] == "telescope.diameter_m")
    assert record["status"] == "assumed"
