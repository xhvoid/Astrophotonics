"""Install a built wheel in a temporary directory and run its public CLI there."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    """Verify wheel imports and noiseless baseline independently of the src checkout."""
    root = Path(__file__).resolve().parents[1]
    wheels = sorted((root / "dist").glob("aophoton-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Build exactly one wheel in dist/ with python -m pip wheel --no-deps . -w dist")
    wheel = wheels[0]
    with tempfile.TemporaryDirectory(prefix="aophoton-wheel-") as temporary:
        target = Path(temporary) / "installed"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                        "--target", str(target), str(wheel)], check=True, capture_output=True, text=True)
        env = dict(os.environ, PYTHONPATH=str(target), MPLCONFIGDIR=str(root / ".mplconfig"))
        location = subprocess.run([sys.executable, "-c", "import aophoton; print(aophoton.__file__)"],
                                  cwd=temporary, env=env, check=True, capture_output=True, text=True)
        assert Path(location.stdout.strip()).is_relative_to(target), "Imported source checkout, not wheel"
        output = Path(temporary) / "observation"
        subprocess.run([sys.executable, "-m", "aophoton", "simulate", "--config",
                        str(root / "configs/baseline.yaml"), "--output", str(output), "--no-noise"],
                       cwd=temporary, env=env, check=True, capture_output=True, text=True)
        summary = json.loads((output / "summary.json").read_text())
        assert abs(summary["recovery_bias_m_s"]) < 0.01
        report = dict(status="passed", wheel=wheel.name,
                      sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      imported_from_separate_install=True, cli_outside_checkout="passed",
                      noiseless_rv_error_m_s=summary["recovery_bias_m_s"],
                      dependency_check="Existing validated environment supplies dependencies; wheel installed with --no-deps")
    (root / "results/package_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
