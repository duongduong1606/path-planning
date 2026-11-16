"""Safely export the official UFLD-v2 TuSimple checkpoint to ONNX."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import types

import torch


OFFICIAL_TUSIMPLE_RES18_SHA256 = "490d8f81995cf6382005e037ca70b5260de6cff7da863c0965b2daeb3f710a4c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("third_party/Ultra-Fast-Lane-Detection-v2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/tusimple_res18.pth"))
    parser.add_argument("--output", type=Path, default=Path("models/tusimple_res18.onnx"))
    parser.add_argument("--validate-only", action="store_true",
                        help="load checkpoint and run one forward pass without the slow ONNX export")
    args = parser.parse_args()
    actual_hash = sha256(args.checkpoint)
    if actual_hash != OFFICIAL_TUSIMPLE_RES18_SHA256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {actual_hash}")
    print("Checkpoint hash verified", flush=True)
    sys.path.insert(0, str(args.source.resolve()))
    from utils.config import Config

    # The official model file imports initialize_weights from utils.common;
    # utils.common eagerly imports NVIDIA DALI even though export does not use
    # a data loader. Supply only that small initialization API so conversion
    # remains independent of DALI.
    common_stub = types.ModuleType("utils.common")
    def initialize_weights(*models):
        # Every parameter is replaced immediately by the strict checkpoint
        # load below. Initializing the 350+ MB classifier first wastes minutes.
        return None
    common_stub.initialize_weights = initialize_weights
    sys.modules["utils.common"] = common_stub
    from model.model_culane import parsingNet

    config = Config.fromfile(str(args.source / "configs/tusimple_res18.py"))
    config.batch_size = 1
    # Skip default initialization of the huge fully-connected head; the strict
    # checkpoint load immediately replaces it.
    original_linear_reset = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda self: None
    try:
        model = parsingNet(
            pretrained=False, backbone=config.backbone,
            num_grid_row=config.num_cell_row, num_cls_row=config.num_row,
            num_grid_col=config.num_cell_col, num_cls_col=config.num_col,
            num_lane_on_row=config.num_lanes, num_lane_on_col=config.num_lanes,
            use_aux=config.use_aux, input_height=config.train_height,
            input_width=config.train_width, fc_norm=config.fc_norm,
        )
    finally:
        torch.nn.Linear.reset_parameters = original_linear_reset
    print("Model graph constructed without classifier initialization", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print("Checkpoint tensors loaded", flush=True)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"checkpoint/model mismatch: {incompatible}")
    print("Checkpoint keys matched", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model moved to {device}", flush=True)
    example = torch.zeros(1, 3, config.train_height, config.train_width, device=device)
    with torch.no_grad():
        result = model(example)
    expected = {
        "loc_row": (1, 100, 56, 4), "loc_col": (1, 100, 41, 4),
        "exist_row": (1, 2, 56, 4), "exist_col": (1, 2, 41, 4),
    }
    shapes = {name: tuple(value.shape) for name, value in result.items()}
    if shapes != expected:
        raise ValueError(f"unexpected UFLD-v2 output contract: {shapes}")
    print(f"Checkpoint valid on {device}: {shapes}")
    if args.validate_only:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model, example, args.output,
            input_names=["input"],
            output_names=["loc_row", "loc_col", "exist_row", "exist_col"],
            opset_version=17, dynamo=False, do_constant_folding=False,
        )
    print(f"Exported {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

