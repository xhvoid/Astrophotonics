"""Command line entry point; paths are relative to the caller's working directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .simulation import run_simulation, save_result


def main(argv: list[str] | None = None) -> int:
    """Run one configured simulation and export its inspectable numerical products."""
    parser = argparse.ArgumentParser(description="Independent AO–photonic spectrograph conceptual simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("simulate", help="Run a seeded synthetic observation")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("results/baseline"))
    run.add_argument("--no-noise", action="store_true", help="Return expected counts for injection/recovery validation")
    args = parser.parse_args(argv)
    try:
        result = run_simulation(load_config(args.config), noisy=not args.no_noise)
        directory = save_result(result, args.output)
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"aophoton: {exc}\n")
    print(json.dumps(result.summary(), indent=2, allow_nan=False))
    print(f"Outputs: {directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

