"""Decode controlsState.enabled from raw logs into causal alignment caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from prepare_lanes import segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    parser.add_argument("--openpilot-root", type=Path, required=True,
                        help="compatible openpilot checkout containing tools.lib.logreader")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.openpilot_root.resolve()))
    try:
        from tools.lib.logreader import LogReader
    except ImportError as error:
        raise SystemExit("Could not import tools.lib.logreader from --openpilot-root") from error
    failures = []
    all_segments = segments(args.data_root)
    for number, segment in enumerate(all_segments, 1):
        destination = segment / "controls_enabled.npz"
        if destination.exists() and not args.overwrite:
            continue
        times, enabled = [], []
        try:
            for message in LogReader(str(segment / "raw_log.bz2")):
                if message.which() == "controlsState":
                    times.append(message.logMonoTime * 1e-9)
                    enabled.append(bool(message.controlsState.enabled))
            if not times:
                raise ValueError("no controlsState messages")
            np.savez_compressed(destination, t=np.asarray(times, np.float64),
                                enabled=np.asarray(enabled, bool))
            print(f"[{number}/{len(all_segments)}] {segment}: {len(times)} states")
        except Exception as error:  # preserve unknown rather than calling it manual
            failures.append(f"{segment}: {error}")
    if failures:
        print("Decode failures (these segments remain control_source=unknown):")
        print("\n".join(failures))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

