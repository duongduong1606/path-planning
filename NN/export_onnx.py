"""Export the exact TCN runtime contract for MATLAB/Simulink."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pipeline_config import FEATURE_NAMES, PipelineConfig
from tcn_model import LateralPlanningTCN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--normalization", type=Path, default=Path("data/processed/normalization.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lateral_tcn.onnx"))
    args = parser.parse_args()
    model = LateralPlanningTCN()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 40, 16), args.output,
                      input_names=["history"], output_names=["terminal_state"],
                      dynamic_axes={"history": {0: "batch"}, "terminal_state": {0: "batch"}},
                      opset_version=17, dynamo=False)
    contract = {"input_shape": ["batch", 40, 16], "feature_names": FEATURE_NAMES,
                "output_names": ["y_T_m", "y_dot_T_mps", "y_ddot_T_mps2", "log_sigma_y"],
                "horizon_seconds": PipelineConfig().horizon_seconds,
                "normalization": json.loads(args.normalization.read_text())}
    args.output.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

