# Local lateral-planning pipeline

This directory contains a reproducible comma2k19 pipeline for predicting the
2.5 s ego-frame terminal state:

```text
[y_T, y_dot_T, y_ddot_T, log_sigma_y]
```

The runtime input is a causal `40 x 16` buffer sampled at 20 Hz. Lane geometry
is an input/constraint; future global pose transformed into the current ego
frame is the behavioral ground truth.

## Summary

- **Goal** — predict the 2.5 s future ego lateral state
  `[y_T, y_dot_T, y_ddot_T, log_sigma_y]` from 2 s of history, learning from human
  driving only (comma2k19 future pose as truth).
- **Approach** — comma2k19 clips → Polar R-CNN lane perception (CULane ResNet18
  checkpoint, fine-tuned on 236 comma2k19 frames) → 16-feature buffer → causal
  TCN (87,508 parameters, receptive field 61) with a Gaussian NLL head → model-only
  acceptance gate on a route-level split. Bounded Frenet-quintic projection and
  Openpilot/expert-teacher distillation are reported as separate diagnostics.
- **Progress** — pipeline, training, evaluation, ONNX/MATLAB export and 5 test
  modules are complete. Test split: lateral FDE **0.29 m** vs 0.85 m for the
  lane-centre quintic (−66 %), coverage90 92.7 %, boundary violation 5.5 %.
  The gate still **fails** "no boundary-violation regression" (0.5 % baseline);
  the offline shield lowers it to 0.4 % but is excluded from acceptance. The
  model-only MATLAB closed loop departs the lane on curve/recovery scenarios, so
  the network is not yet accepted as a standalone tracker.
- **Next** — lane-loss-aware boundary regression, intent/object inputs, and a
  model-only closed-loop acceptance run.
- **Perception audit** — 290 annotated frames: both-lane recall 0.75, median
  pixel error 10.4 px, p95 65.7 px; perception is the current bottleneck.

## Install

Use a dedicated CUDA environment for lane inference and training:

```powershell
python -m pip install -r requirements.txt
```

Lane perception uses the official Polar R-CNN ResNet18 CULane checkpoint at
`models/polarrcnn_culane_r18.pth` (SHA-256
`7c537ab1dded83ea9bace02d3b6f79d6b44e36b2768f4fe919a47996a540dd05`).
The upstream source is kept at `third_party/PolarRCNN`. Its NMS-free inference
path does not require compiling the optional CUDA NMS extension.

The comma2k19-adapted checkpoint selected at epoch 15 is stored at
`models/polarrcnn_comma_finetuned_e15.pth` (SHA-256
`63d0ae1651f024b7a40b9bda05490c4bfe6bbeac62d994a73b611a3c05600e8b`).
It freezes the ResNet18 backbone and fine-tunes the FPN/RPN/ROI heads on the
route-separated manual audit set (236/23/31 train/validation/test frames).

## End-to-end workflow

From `C:\Users\PC\Path_planning\NN`:

```powershell
# Optional but required for a claim of human-only learning.
python decode_controls.py --openpilot-root C:\path\to\compatible\openpilot

# Polar R-CNN inference, resumable per segment, then route calibration.
# Fine-tuned results go to data/lane_cache_polarrcnn_ft_e15, separate from
# the original Polar R-CNN and UFLD caches.
python prepare_lanes.py --model models\polarrcnn_comma_finetuned_e15.pth

# Recreate the small domain-adaptation experiment when annotations change.
python prepare_polarrcnn_finetune.py
python finetune_polarrcnn.py --epochs 20

# Deterministic route split, synchronization, truth and HDF5 generation.
python build_dataset.py

# Generate Openpilot teacher caches separately from immutable pose truth.
# Start with a short recurrent smoke test, then audit coordinate sign/scale.
python prepare_teacher.py generate --limit-segments 1 --max-frames 200
python prepare_teacher.py audit

# After full teacher generation and audit, enable secondary distillation.
# Validation/model selection still uses human pose truth only.
python train_model.py --model tcn --teacher-loss-weight 0.10

# Distillation defaults to terminal lateral position only. Openpilot plan
# velocity/acceleration are not treated as equivalent to pose-fit derivatives.

# Alternative teacher: minimally project human targets into a constrained
# Frenet/quintic feasible set. This is offline supervision, not a shield.
python prepare_expert_teacher.py generate
python prepare_expert_teacher.py audit
python train_model.py --model tcn --epochs 10 --learning-rate 1e-4 `
  --initial-checkpoint artifacts/tcn_lane_safe/tcn.pt `
  --teacher-cache data/teacher_frenet_expert --teacher-loss-weight 0.10 `
  --teacher-max-pose-disagreement 4.0 `
  --teacher-velocity-weight 0.5 --teacher-acceleration-weight 0.25 `
  --output artifacts/tcn_frenet_expert

# Evaluate raw model output, then make a paired A/B decision against model A.
python run_evaluation.py artifacts/tcn_frenet_expert/tcn_test_predictions.npz `
  --output artifacts/tcn_frenet_expert/tcn_metrics.json
python compare_experiments.py `
  artifacts/tcn_lane_safe/tcn_test_predictions.npz `
  artifacts/tcn_frenet_expert/tcn_test_predictions.npz `
  --output artifacts/tcn_frenet_expert/ab_vs_lane_safe.json

# Create/score the independent 300-frame perception audit.
python lane_audit.py sample
python lane_audit.py annotate data/lane_audit/annotations.csv
# Annotator keys: U undo, R reset current frame, B reopen previous frame,
# S skip, Q save progress and quit.
python lane_audit.py evaluate data/lane_audit/annotations.csv --output artifacts/lane_audit.json
python quality_gate.py --lane-audit artifacts/lane_audit.json

# Train and compare all learned models.
python train_model.py --model tcn --lane-loss-weight 1.0 --lane-margin 0.20
# Or fine-tune the existing behavioral checkpoint with the new safety loss:
python train_model.py --model tcn --epochs 10 --learning-rate 3e-4 `
  --lane-loss-weight 1.0 --lane-margin 0.20 `
  --initial-checkpoint artifacts/tcn.pt --output artifacts/tcn_lane_safe
python train_model.py --model mlp
python train_model.py --model gru
python run_evaluation.py artifacts/tcn_test_predictions.npz --output artifacts/tcn_metrics.json `
  --projected-output artifacts/tcn_test_predictions_projected.npz

# Export for MATLAB/Simulink after offline acceptance.
python export_onnx.py artifacts/tcn.pt
python export_parity_fixture.py
```

`prepare_lanes.py --backend classical --allow-default-calibration` exists only
for smoke tests and geometric baseline work. UFLD-v2 remains available with
`--backend ufldv2`, but neither cache is silently reused for Polar R-CNN.

## Fixed contracts

- Features: left/right cubic coefficients `c0..c3`, two confidences, speed,
  yaw rate, lateral acceleration, steering angle/rate, perception validity.
- Polynomial: `y(x)=c0+c1*x+c2*x^2+c3*x^3`, x forward and y left, metres.
- HDF5: one file per segment, plus `manifest.csv`, `pipeline_config.json`, and
  train-only `normalization.json`.
- Split: 26/5/6 routes for train/validation/test with seed 2026; no window split.
- Manual window: all samples from the start of the 2 s history through the end
  of the 2.5 s future have decoded lateral automation disabled. Unknown control
  state is never relabeled as manual.
- Model: causal TCN channels `[32,48,64,64]`, dilations `[1,2,4,8]`, kernel 3,
  two convolutions/block, receptive field 61, head `[128,64,32]`, 87,508
  trainable parameters.
- Training adds a confidence-weighted lane-feasibility loss on the reconstructed
  quintic path. Model acceptance always uses raw network predictions. The
  bounded projection in `safety_shield.py` is reported separately as a future
  shield diagnostic and cannot make a failing AI planner pass its gate.
- Teacher: official Openpilot `big_driving_supercombo` ONNX, run offline with
  the exact narrow/wide virtual-camera warp, I420 packing and recurrent state.
  Teacher caches live under `data/teacher_openpilot`; they never overwrite
  future-pose ground truth. The default desire is zero/no-request, so the
  teacher does not invent lane-change labels absent from the recorded data.
  Its right-positive lateral output is converted to the project's left-positive
  convention (`lateral_sign=-1`), validated against future pose before use.

Run the test suite with:

```powershell
python -m pytest -q -p no:cacheprovider
```

MATLAB parity is checked after building the full dataset:

```matlab
addpath('matlab');
validate_preprocessing_parity('artifacts/parity');
```

## Model-only closed-loop scenarios

The MATLAB runner reuses the preview controller, steering-rate limit and
kinematic bicycle equations from `Highway Lane Change Planner and Controller`.
It compares raw TCN decisions with a Frenet lane-center baseline; the safety
shield is intentionally excluded from model acceptance.

```powershell
python export_matlab_tcn.py artifacts\tcn_lane_safe\tcn.pt
```

```matlab
addpath('matlab');
validate_tcn_parity('artifacts/tcn_lane_safe/lateral_tcn_weights.mat');
run_tcn_closed_loop_scenarios;
```

Outputs are written to `artifacts/closed_loop`: a CSV/JSON summary, complete
MATLAB traces and a scenario comparison figure. The scenarios cover nominal
straight/curved roads, lateral/heading recovery, deterministic departure and
low lane confidence. Object avoidance and lane-change initiation are excluded
because the 16-feature TCN has no object or intent input.
