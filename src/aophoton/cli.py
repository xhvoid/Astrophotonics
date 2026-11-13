"""Command line entry point; paths are relative to the caller's working directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .config import config_from_dict, load_config
from .simulation import file_sha256, run_simulation, save_result

# Scientific results in summary.json that a replay must reproduce, by comparison
# rule.  Flags, counts, labels and configured values must be equal.
REPLAY_EXACT = ("seed", "feed", "injected_rv_m_s", "detector_pixels_per_spectral_sample", "rv_at_search_boundary")
# Floating-point results.  The same code on the same inputs differs only by
# roundoff (summation order, BLAS threading), many orders below rtol 1e-9.
REPLAY_FLOAT = ("residual_wfe_nm", "measured_strehl", "marechal_without_jitter", "mean_smf_coupling",
                "mean_few_mode_coupling", "mean_lantern_coupling", "mean_selected_coupling", "collecting_area_m2",
                "median_snr_per_spectral_sample", "median_snr_per_pixel", "rv_reduced_chi2")
# Fitted velocities also carry the bounded refinement's convergence tolerance
# (xatol = 1e-5 m/s in recover_rv), allowed three times over.
REPLAY_VELOCITY = ("recovered_rv_m_s", "recovery_bias_m_s")
VELOCITY_ATOL_M_S = 3e-5
# The Fisher sigma is evaluated at the fitted RV and moves with it.  By
# Cauchy-Schwarz its relative change is at most |delta RV| divided by the
# gradient-to-curvature velocity scale of velocity_correlation_scale, which
# LSF broadening keeps above about 0.8 LSF sigma (1 km/s at R = 100,000; 100 m/s
# even at R = 1,000,000): 3e-5 m/s then moves sigma by at most 3e-7.
REPLAY_UNCERTAINTY = ("estimated_rv_uncertainty_m_s",)
UNCERTAINTY_RTOL = 3e-7
REPLAYED_RESULTS = REPLAY_EXACT + REPLAY_FLOAT + REPLAY_VELOCITY + REPLAY_UNCERTAINTY
# Not compared: rv_model_evaluations (solver cost, not an observable; a roundoff
# change in the refinement path can alter it by a few calls without changing the
# fit) and descriptive text (disclaimer, definitions, column descriptions).


def _differs(key: str, recorded, replayed) -> bool:
    if key in REPLAY_EXACT:
        return type(recorded) is not type(replayed) or recorded != replayed
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
        return True
    if key in REPLAY_VELOCITY:
        return not np.isclose(replayed, recorded, rtol=0.0, atol=VELOCITY_ATOL_M_S)
    if key in REPLAY_UNCERTAINTY:
        return not np.isclose(replayed, recorded, rtol=UNCERTAINTY_RTOL, atol=0.0)
    return not np.isclose(replayed, recorded, rtol=1e-9, atol=1e-12)


def _replay(summary_path: Path):
    """Rebuild a recorded run: same configuration, noise mode and template content.

    Returns the replayed result and ``[(field, recorded, replayed), ...]`` for
    every scientific result in ``REPLAYED_RESULTS`` that differs.  Nothing is
    written.
    """
    recorded = json.loads(Path(summary_path).read_text())
    for key in ("configuration", "noise_mode", "template", "run_identity_sha256"):
        if key not in recorded:
            raise ValueError(f"{summary_path} has no {key!r}; it predates replayable summaries.")
    missing = [key for key in REPLAYED_RESULTS if key not in recorded]
    if missing:
        raise ValueError(f"{summary_path} lacks the replayed results {missing}; it was written by an older "
                         "summary schema and cannot be checked completely.")
    if recorded["noise_mode"] not in ("noisy", "noiseless"):
        raise ValueError(f"Unknown noise_mode {recorded['noise_mode']!r} in {summary_path}.")
    config = config_from_dict(recorded["configuration"])
    template = recorded["template"]
    if template.get("source") == "file" and file_sha256(template["path"]) != template["file_sha256"]:
        raise ValueError(f"Template {template['path']} no longer matches the recorded SHA-256 "
                         f"{template['file_sha256']}; refusing to replay a different template.")
    result = run_simulation(config, noisy=recorded["noise_mode"] == "noisy")
    if result.template != template:
        raise ValueError("Template samples differ from the recorded run; refusing to replay it.")
    if result.run_identity() != recorded["run_identity_sha256"]:
        raise ValueError("Replayed run identity differs from the recorded run_identity_sha256.")
    summary = result.summary()
    mismatched = [(key, recorded[key], summary[key]) for key in REPLAYED_RESULTS
                  if _differs(key, recorded[key], summary[key])]
    return result, mismatched


def _same_directory(path: Path, other: Path) -> bool:
    """Whether two paths name one existing directory (aliases and links included)."""
    try:
        return path.exists() and os.path.samefile(path, other)
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    """Run or replay one configured simulation and export its inspectable numerical products.

    A replay writes nothing unless every scientific result reproduces, so a
    failed replay leaves its reference and any existing destination untouched.
    A successful replay into the reference's own directory leaves it unchanged.
    """
    parser = argparse.ArgumentParser(description="Independent AO–photonic spectrograph conceptual simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("simulate", help="Run a seeded synthetic observation")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("results/baseline"))
    run.add_argument("--no-noise", action="store_true", help="Return expected counts for injection/recovery validation")
    replay = commands.add_parser("replay", help="Rerun a saved summary.json and check it reproduces")
    replay.add_argument("--summary", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "replay":
            result, mismatched = _replay(args.summary)
            if mismatched:
                print(f"Replay of {args.summary} ({result.noise_mode}) differs in {len(mismatched)} of "
                      f"{len(REPLAYED_RESULTS)} scientific results:")
                for key, recorded, replayed in mismatched:
                    print(f"  {key}: recorded {recorded!r}, replayed {replayed!r}")
                print(f"Nothing was written; {args.summary} and {args.output} are unchanged.")
                return 1
            if _same_directory(args.output, args.summary.resolve().parent):
                print(f"Replay of {args.summary} ({result.noise_mode}) reproduces all {len(REPLAYED_RESULTS)} "
                      f"checked scientific results; {args.output} is the reference directory, left unchanged.")
                return 0
        else:
            result = run_simulation(load_config(args.config), noisy=not args.no_noise)
        directory = save_result(result, args.output)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(2, f"aophoton: {exc}\n")
    print(json.dumps(result.summary(), indent=2, allow_nan=False))
    print(f"Outputs: {directory.resolve()}")
    if args.command == "replay":
        print(f"Replay of {args.summary} ({result.noise_mode}) reproduces all {len(REPLAYED_RESULTS)} checked "
              "scientific results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
