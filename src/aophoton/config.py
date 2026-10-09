"""Strict configuration and provenance; units are encoded in field names."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .detector import DetectorConfig
from .radial_velocity import FISHER_DERIVATIVE_STEP_M_S
from .spectrograph import SpectrographConfig


@dataclass(frozen=True)
class TelescopeConfig:
    """Pupil diameter and spider width in metres; obscuration is a diameter ratio."""

    diameter_m: float = 3.6
    obscuration: float = 0.25
    spider_width_m: float = 0.0
    n_pix: int = 128
    pad_factor: int = 4


@dataclass(frozen=True)
class AOConfig:
    """WFE in nm at reference seeing; jitter is independent Gaussian RMS per axis."""

    wavelength_um: float = 1.555
    seeing_arcsec: float = 0.8
    reference_seeing_arcsec: float = 0.8
    residual_wfe_nm: float = 150.0
    outer_scale_m: float = 25.0
    tip_tilt_rms_mas: float = 10.0
    n_frames: int = 12
    wavelength_nodes: int = 3


@dataclass(frozen=True)
class FibreConfig:
    """Fixed angular Gaussian 1/e field radius; equivalent MFD = twice this value."""

    mode_radius_mas: float = 57.0
    few_mode_count: int = 6
    lantern_ports: int = 6
    lantern_transmission: float = 0.9
    feed: str = "smf"


@dataclass(frozen=True)
class ObservationConfig:
    """Synthetic velocity in m/s; optional template path resolved relative to YAML."""

    injected_rv_m_s: float = 2300.0
    rv_search_half_width_m_s: float = 10000.0
    template_path: str | None = None
    template_seed: int = 7


@dataclass(frozen=True)
class SimulationConfig:
    """One reproducible independent conceptual observation."""

    seed: int = 2026
    telescope: TelescopeConfig = field(default_factory=TelescopeConfig)
    ao: AOConfig = field(default_factory=AOConfig)
    fibre: FibreConfig = field(default_factory=FibreConfig)
    spectrograph: SpectrographConfig = field(default_factory=SpectrographConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def __post_init__(self) -> None:
        values = asdict(self)
        for group, entries in values.items():
            for name, value in (entries.items() if isinstance(entries, dict) else [(group, entries)]):
                if isinstance(value, (int, float)) and (isinstance(value, bool) or not np.isfinite(value)):
                    raise ValueError(f"{group}.{name} must be finite and numeric")
        for name, value in [("seed", self.seed), ("n_pix", self.telescope.n_pix),
                            ("pad_factor", self.telescope.pad_factor), ("n_frames", self.ao.n_frames),
                            ("wavelength_nodes", self.ao.wavelength_nodes),
                            ("few_mode_count", self.fibre.few_mode_count),
                            ("lantern_ports", self.fibre.lantern_ports),
                            ("template_seed", self.observation.template_seed)]:
            if not isinstance(value, int) or value < (0 if "seed" in name else 1):
                raise ValueError(f"{name} must be an integer in its valid range")
        t, a, f, o = self.telescope, self.ao, self.fibre, self.observation
        if t.diameter_m <= 0 or not 0 <= t.obscuration < 1 or t.spider_width_m < 0:
            raise ValueError("Invalid pupil dimensions")
        if t.n_pix < 32 or t.pad_factor < 2:
            raise ValueError("Use at least 32 pupil samples and padding factor 2")
        if min(a.wavelength_um, a.seeing_arcsec, a.reference_seeing_arcsec, a.outer_scale_m) <= 0:
            raise ValueError("Wavelength, seeing, and outer scale must be positive")
        if a.residual_wfe_nm < 0 or a.tip_tilt_rms_mas < 0:
            raise ValueError("WFE and jitter must be nonnegative")
        if f.mode_radius_mas <= 0 or not 0 <= f.lantern_transmission <= 1:
            raise ValueError("Invalid mode radius or lantern transmission")
        if f.feed not in {"smf", "few_mode", "lantern"}:
            raise ValueError("feed must be smf, few_mode, or lantern")
        if a.wavelength_nodes < 2:
            raise ValueError("At least two wavelength nodes are required for spectral coupling")
        if not self.spectrograph.wavelength_min_um <= a.wavelength_um <= self.spectrograph.wavelength_max_um:
            raise ValueError("AO reference wavelength must lie inside the spectral band")
        # The Fisher derivative samples RV +/- its step, so a best fit at the
        # search edge must still leave that step inside the template padding.
        if not (0 < o.rv_search_half_width_m_s
                and o.rv_search_half_width_m_s + FISHER_DERIVATIVE_STEP_M_S <= self.spectrograph.rv_padding_m_s):
            raise ValueError("RV search must be positive and inside the template padding, "
                             f"leaving the {FISHER_DERIVATIVE_STEP_M_S} m/s Fisher derivative step")
        if abs(o.injected_rv_m_s) >= o.rv_search_half_width_m_s:
            raise ValueError("Injected RV must lie strictly inside the search interval")


def load_config(path: str | Path) -> SimulationConfig:
    """Read YAML, reject misspelled/unknown keys, and resolve template paths."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    known = {f.name for f in fields(SimulationConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    constructors = dict(telescope=TelescopeConfig, ao=AOConfig, fibre=FibreConfig,
                        spectrograph=SpectrographConfig, detector=DetectorConfig,
                        observation=ObservationConfig)
    kwargs: dict[str, Any] = {}
    for group, cls in constructors.items():
        data = raw.get(group, {})
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown keys or invalid mapping in {group}")
        if group == "observation" and data.get("template_path") is not None:
            data["template_path"] = str((path.parent / data["template_path"]).resolve())
        kwargs[group] = cls(**data)
    return SimulationConfig(seed=raw.get("seed", 2026), **kwargs)


def parameter_records(config: SimulationConfig) -> list[dict[str, Any]]:
    """Return provenance for EVERY setting; altered sourced defaults become assumed."""
    sourced = {"telescope.diameter_m": 3.6, "spectrograph.resolving_power": 100000.0,
               "spectrograph.pixels_per_resolution_element": 3.0}
    source = "https://www.eso.org/sci/facilities/develop/instruments/NIRPS.html"
    records = []
    for group, value in asdict(config).items():
        entries = value.items() if isinstance(value, dict) else [(None, value)]
        for key, val in entries:
            name = f"{group}.{key}" if key else group
            is_sourced = name in sourced and val == sourced[name]
            illustrative = name.startswith("observation.") or name == "seed" or name.endswith(("n_pix", "pad_factor", "n_frames", "wavelength_nodes", "oversample"))
            records.append(dict(parameter=name, value=val,
                                status="sourced" if is_sourced else "illustrative" if illustrative else "assumed",
                                source=source if is_sourced else "User-selected conceptual model; no instrument calibration",
                                note="Source-inspired design value; not a claim of current instrument performance" if is_sourced else "No fitted instrument parameters"))
    return records

