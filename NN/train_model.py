"""Reproducible training for TCN, flattened-MLP and GRU experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HDF5WindowDataset, choose_training_policy
from models import (
    FlattenedMLP, GRUBaseline, teacher_distillation_loss,
    terminal_heteroscedastic_loss,
)
from tcn_model import LateralPlanningTCN


def make_model(name: str) -> torch.nn.Module:
    return {"tcn": LateralPlanningTCN, "mlp": FlattenedMLP, "gru": GRUBaseline}[name]()


def run_epoch(model, loader, optimizer, device: torch.device, args) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total = count = teacher_used = 0
    mean = torch.as_tensor(loader.dataset.mean, device=device)
    std = torch.as_tensor(loader.dataset.std, device=device)
    for features, target, metadata in loader:
        features, target = features.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            prediction = model(features)
            physical_current = features[:, -1] * std + mean
            loss = terminal_heteroscedastic_loss(
                prediction, target, physical_current,
                lane_weight=args.lane_loss_weight,
                lane_margin_m=args.lane_margin,
            )
            if training and args.teacher_loss_weight:
                teacher_target = metadata["teacher_target"].to(device)
                teacher_std = metadata["teacher_std"].to(device)
                teacher_valid = metadata["teacher_valid"].to(device)
                # Keep supervision inside the student's trustworthy lane
                # domain and reject abnormal teacher/human disagreement.
                teacher_valid &= physical_current[:, 8:10].amin(dim=1) >= args.teacher_min_lane_confidence
                teacher_valid &= physical_current[:, 15] > 0
                teacher_valid &= teacher_std[:, 0] <= args.teacher_max_position_std
                teacher_valid &= torch.abs(teacher_target[:, 0] - target[:, 0]) <= args.teacher_max_pose_disagreement
                loss = loss + args.teacher_loss_weight * teacher_distillation_loss(
                    prediction, teacher_target, teacher_std, teacher_valid,
                    component_weights=(1.0, args.teacher_velocity_weight,
                                       args.teacher_acceleration_weight),
                )
                teacher_used += int(teacher_valid.sum())
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        total += float(loss)*len(features); count += len(features)
    return total/max(count, 1), teacher_used/max(count, 1)


def train_stage(model, dataset, validation, epochs, args, device, checkpoint: Path) -> None:
    generator = torch.Generator().manual_seed(2026)
    train_loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation, args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(1, epochs+1):
        train_loss, teacher_fraction = run_epoch(model, train_loader, optimizer, device, args)
        validation_loss, _ = run_epoch(model, validation_loader, None, device, args)
        print(f"epoch={epoch} train={train_loss:.6f} validation={validation_loss:.6f} "
              f"teacher_eligible={teacher_fraction:.3f}")
        if validation_loss < best:
            best = validation_loss
            torch.save({"model": model.state_dict(), "validation_loss": best,
                        "model_name": args.model, "seed": 2026,
                        "lane_loss_weight": args.lane_loss_weight,
                        "lane_margin_m": args.lane_margin,
                        "teacher_loss_weight": args.teacher_loss_weight,
                        "teacher_min_lane_confidence": args.teacher_min_lane_confidence,
                        "teacher_max_position_std": args.teacher_max_position_std,
                        "teacher_max_pose_disagreement": args.teacher_max_pose_disagreement,
                        "teacher_velocity_weight": args.teacher_velocity_weight,
                        "teacher_acceleration_weight": args.teacher_acceleration_weight}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])


def save_predictions(model, dataset, output: Path, device: torch.device, batch_size: int) -> None:
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0)
    predictions, keys = [], []
    model.eval()
    with torch.no_grad():
        for features, _, metadata in loader:
            predictions.append(model(features.to(device)).cpu().numpy())
            keys.extend(f"{path}::{int(anchor)}" for path, anchor in
                        zip(metadata["hdf5"], metadata["anchor"]))
    np.savez_compressed(output, keys=np.asarray(keys), prediction=np.concatenate(predictions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", choices=("tcn", "mlp", "gru"), default="tcn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lane-loss-weight", type=float, default=1.0)
    parser.add_argument("--lane-margin", type=float, default=0.20)
    parser.add_argument("--teacher-cache", type=Path, default=Path("data/teacher_openpilot"))
    parser.add_argument("--teacher-loss-weight", type=float, default=0.0,
                        help="secondary offline teacher weight; 0 disables teacher")
    parser.add_argument("--teacher-min-lane-confidence", type=float, default=0.50)
    parser.add_argument("--teacher-max-position-std", type=float, default=0.50)
    parser.add_argument("--teacher-max-pose-disagreement", type=float, default=0.75)
    parser.add_argument("--teacher-velocity-weight", type=float, default=0.0,
                        help="keep 0: Openpilot lateral velocity is not pose-label equivalent")
    parser.add_argument("--teacher-acceleration-weight", type=float, default=0.0,
                        help="keep 0 unless a separate dynamics parity audit passes")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="optional checkpoint used to initialize the model")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / f"{args.model}.pt"
    policy = choose_training_policy(args.processed)
    print(f"device={device} policy={policy}")
    model = make_model(args.model).to(device)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
        if initial.get("model_name") not in (None, args.model):
            raise ValueError(
                f"checkpoint model {initial.get('model_name')!r} does not match {args.model!r}"
            )
        model.load_state_dict(initial["model"])
        print(f"initialized_from={args.initial_checkpoint}")
    validation = HDF5WindowDataset(args.processed, "validation")
    teacher_root = args.teacher_cache if args.teacher_loss_weight else None
    if policy == "manual_only":
        train_stage(model, HDF5WindowDataset(args.processed, "train", manual_only=True,
                                             teacher_root=teacher_root),
                    validation, args.epochs, args, device, checkpoint)
    else:
        train_stage(model, HDF5WindowDataset(args.processed, "train", teacher_root=teacher_root),
                    validation, args.epochs, args, device, checkpoint)
        if policy == "pretrain_all_then_finetune_manual":
            train_stage(model, HDF5WindowDataset(args.processed, "train", manual_only=True,
                                                 teacher_root=teacher_root),
                        validation, args.fine_tune_epochs, args, device, checkpoint)
    save_predictions(model, HDF5WindowDataset(args.processed, "test"),
                     args.output / f"{args.model}_test_predictions.npz", device, args.batch_size)


if __name__ == "__main__":
    main()

