"""Public configuration, deterministic end-to-end chain, and CLI output contract."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from aophoton.cli import main
from aophoton.config import load_config, parameter_records
from aophoton.simulation import run_simulation, save_result

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
