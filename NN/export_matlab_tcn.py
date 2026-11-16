"""Export a PyTorch TCN checkpoint for dependency-free MATLAB inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
import torch

from tcn_model import LateralPlanningTCN


def matlab_name(name: str) -> str:
    return name.replace(".", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--normalization", type=Path,
                        default=Path("data/processed/normalization.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/tcn_lane_safe/lateral_tcn_weights.mat"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = LateralPlanningTCN()
    model.load_state_dict(state)
    model.eval()

    rng = np.random.default_rng(2026)
    parity_input = rng.normal(0.0, 0.75, size=(40, 16)).astype(np.float32)
    with torch.no_grad():
        parity_expected = model(torch.from_numpy(parity_input)[None]).numpy()[0]

    weights = {matlab_name(name): value.detach().cpu().numpy()
               for name, value in state.items()}
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    payload = {
        "weights": weights,
        "normalization": {
            "clip_low": np.asarray(normalization["clip_low"], np.float64),
            "clip_high": np.asarray(normalization["clip_high"], np.float64),
            "mean": np.asarray(normalization["mean"], np.float64),
            "std": np.asarray(normalization["std"], np.float64),
        },
        "parity_input": parity_input.astype(np.float64),
        "parity_expected": parity_expected.astype(np.float64),
        "sample_rate_hz": 20.0,
        "history_steps": 40,
        "horizon_seconds": 2.5,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.output, payload, do_compression=True, long_field_names=True)
    print(args.output)


if __name__ == "__main__":
    main()
