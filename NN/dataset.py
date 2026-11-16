"""Windowed HDF5 dataset with train-only normalization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5WindowDataset(Dataset):
    def __init__(
        self, root: Path, split: str, manual_only: bool | None = None,
        teacher_root: Path | None = None,
    ) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError as error:
            raise RuntimeError("h5py is required") from error
        self.root = Path(root)
        with (self.root / "manifest.csv").open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not rows:
            raise ValueError(f"no segments for split {split!r}")
        normalization = json.loads((self.root / "normalization.json").read_text())
        self.low = np.asarray(normalization["clip_low"], np.float32)
        self.high = np.asarray(normalization["clip_high"], np.float32)
        self.mean = np.asarray(normalization["mean"], np.float32)
        self.std = np.asarray(normalization["std"], np.float32)
        self.history_steps = 40
        self.index: list[tuple[Path, int]] = []
        self.teacher: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        import h5py
        for row in rows:
            path = self.root / row["hdf5"]
            with h5py.File(path, "r") as file:
                if manual_only is True:
                    anchors = file["manual_window_anchors"][:]
                else:
                    anchors = file["window_anchors"][:]
            self.index.extend((path, int(anchor)) for anchor in anchors)
            if teacher_root is not None:
                cache = (Path(teacher_root) / Path(row["hdf5"])).with_suffix(".npz")
                if cache.exists():
                    with np.load(cache, allow_pickle=False) as values:
                        # Retain only the compact terminal teacher contract in
                        # worker memory. It may come from Openpilot or an
                        # offline constrained Frenet expert.
                        self.teacher[row["hdf5"]] = (
                            values["target"].astype(np.float32),
                            values["target_std"].astype(np.float32),
                            values["valid"].astype(bool),
                        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        import h5py
        path, anchor = self.index[item]
        with h5py.File(path, "r") as file:
            values = file["features"][anchor-self.history_steps+1:anchor+1]
            target = file["target"][anchor]
            metadata = {"hdf5": path.relative_to(self.root).as_posix(),
                        "anchor": anchor, "route": file.attrs["route"]}
        key = metadata["hdf5"]
        if key in self.teacher and anchor < len(self.teacher[key][0]):
            teacher_target, teacher_std, teacher_valid = self.teacher[key]
            metadata["teacher_target"] = teacher_target[anchor]
            metadata["teacher_std"] = teacher_std[anchor]
            metadata["teacher_valid"] = bool(teacher_valid[anchor])
        else:
            metadata["teacher_target"] = np.zeros(3, np.float32)
            metadata["teacher_std"] = np.ones(3, np.float32)
            metadata["teacher_valid"] = False
        values = (np.clip(values, self.low, self.high) - self.mean) / self.std
        return torch.from_numpy(values.astype(np.float32)), torch.from_numpy(target), metadata


def choose_training_policy(root: Path, threshold: float = 0.80) -> str:
    """Choose manual-only or all-then-manual without mislabeling unknown control."""
    import h5py
    all_count = manual_count = 0
    with (Path(root) / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "train"]
    for row in rows:
        with h5py.File(Path(root) / row["hdf5"], "r") as file:
            all_count += len(file["window_anchors"])
            manual_count += len(file["manual_window_anchors"])
    fraction = manual_count / max(all_count, 1)
    if fraction >= threshold:
        return "manual_only"
    if manual_count:
        return "pretrain_all_then_finetune_manual"
    return "pretrain_all_manual_unavailable"

