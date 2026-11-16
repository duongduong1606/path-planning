"""Small-data domain adaptation of Polar R-CNN on audited comma2k19 frames."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polar-root", type=Path, default=Path("third_party/PolarRCNN"))
    parser.add_argument("--data-root", type=Path, default=Path("data/polarrcnn_finetune"))
    parser.add_argument("--initial-model", type=Path, default=Path("models/polarrcnn_culane_r18.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/polarrcnn_finetune"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--neck-lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    repository = args.polar_root.resolve()
    sys.path.insert(0, str(repository))
    cfg = importlib.import_module("Config.polarrcnn_culane_r18")
    cfg.pretrained = False
    cfg.data_root = str(args.data_root.resolve())
    cfg.batch_size = args.batch_size
    cfg.random_seed = args.seed
    # Strong enough for a tiny domain set, but less destructive than the
    # original affine scale/translation recipe.
    cfg.train_augments = [
        dict(name="Resize", parameters=dict(height=cfg.img_h, width=cfg.img_w,
                                             interpolation=2, p=1.0)),
        dict(name="HorizontalFlip", parameters=dict(p=0.5)),
        dict(name="RandomBrightnessContrast", parameters=dict(
            brightness_limit=(-0.15, 0.15), contrast_limit=(-0.1, 0.1), p=0.5)),
        dict(name="HueSaturationValue", parameters=dict(
            hue_shift_limit=(-6, 6), sat_shift_limit=(-8, 8), val_shift_limit=(-8, 8), p=0.35)),
        dict(name="Affine", parameters=dict(
            translate_percent=dict(x=(-0.03, 0.03), y=(-0.02, 0.02)),
            rotate=(-3, 3), scale=(0.95, 1.05), interpolation=2, p=0.4)),
        dict(name="Resize", parameters=dict(height=cfg.img_h, width=cfg.img_w,
                                             interpolation=2, p=1.0)),
    ]

    from Dataset.build import build_trainset
    from Loss.overallloss import OverallLoss
    from Models.build import build_model

    dataset = build_trainset(cfg)
    if len(dataset) == 0:
        raise RuntimeError("fine-tune training set is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=dataset.collate_fn, drop_last=False, pin_memory=True)
    model = build_model(cfg)
    state = torch.load(args.initial_model, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    model.cuda()
    loss_function = OverallLoss(cfg).cuda()
    optimizer = torch.optim.AdamW([
        {"params": model.neck.parameters(), "lr": args.neck_lr},
        {"params": model.rpn_head.parameters(), "lr": args.head_lr},
        {"params": model.roi_head.parameters(), "lr": args.head_lr},
    ], weight_decay=1e-4)
    total_steps = args.epochs*len(loader)
    warmup_steps = min(50, max(1, total_steps//10))
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step+1)/warmup_steps
        progress = (step-warmup_steps)/max(total_steps-warmup_steps, 1)
        return 0.5*(1.0+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    step = 0
    for epoch in range(1, args.epochs+1):
        model.train(); model.backbone.eval()
        losses = []
        for batch in loader:
            device_batch = {}
            for key, value in batch.items():
                if key == "plot_img":
                    continue
                device_batch[key] = ([item.cuda(non_blocking=True) for item in value]
                                     if isinstance(value, list) else value.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(device_batch)
            loss, _ = loss_function(prediction, device_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
            optimizer.step(); scheduler.step()
            losses.append(float(loss.detach().cpu())); step += 1
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss,
                        "head_lr": optimizer.param_groups[1]["lr"]})
        print(f"epoch {epoch:02d}/{args.epochs}: loss={mean_loss:.5f}")
        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), args.output/f"epoch_{epoch:02d}.pth")
    metadata = {"train_samples": len(dataset), "epochs": args.epochs,
                "trainable_parameters": trainable, "frozen_parameters": frozen,
                "history": history}
    (args.output/"training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
