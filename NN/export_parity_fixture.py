"""Export deterministic preprocessing values for MATLAB parity validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", type=Path, default=Path("data/processed/normalization.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/parity"))
    args = parser.parse_args()
    stats = json.loads(args.normalization.read_text())
    rng = np.random.default_rng(2026)
    low, high = np.asarray(stats["clip_low"]), np.asarray(stats["clip_high"])
    values = rng.uniform(low-(high-low)*.1, high+(high-low)*.1, size=(40, 16))
    expected = (np.clip(values, low, high)-np.asarray(stats["mean"]))/np.asarray(stats["std"])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output/"input.csv", values, delimiter=",", fmt="%.17g")
    np.savetxt(args.output/"expected.csv", expected, delimiter=",", fmt="%.17g")
    (args.output/"normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

