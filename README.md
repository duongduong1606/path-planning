# Path planning — learned vs. model-based lateral planning

Two independent L2/L2+ lateral-planning prototypes kept in one repository so the
data-driven and the deterministic approach can be compared on the same lane-keeping
task. Each folder is self-contained and has its own README.

| Folder | Goal | Approach | Status |
| --- | --- | --- | --- |
| [`NN/`](NN/README.md) | Learn the driver's 2.5 s future lateral behaviour from comma2k19 | Polar R-CNN lanes → 40 × 16 buffer (2 s @ 20 Hz) → causal TCN + Gaussian NLL head | Trained and evaluated: lateral FDE **0.29 m** (−66 % vs lane-centre quintic), coverage90 92.7 %; acceptance gate still fails the boundary-violation check; closed loop not accepted |
| [`HFSM_ Frenet_quintic_ LTV-MPC/`](HFSM_%20Frenet_quintic_%20LTV-MPC/README.md) | Deterministic, verifiable lateral stack for L2/L2+ | 40 Hz HFSM (`ELK > LKA > HWA`) → Frenet quintic candidates → LTV-MPC (OSQP) → actuator delay + bicycle plant | Baseline KPIs met (18 s run, max lateral error 0.55 m, planner p99 1.05 ms, supervisor interventions 0 %, all quality reserves ≥ 20 %); 23 scenarios and a static UI exporter; C reference planner is a draft |

## Repository layout

```text
README.md                              this file
NN/                                    learned planner (Python, PyTorch, MATLAB export)
HFSM_ Frenet_quintic_ LTV-MPC/          model-based planner (Python package `l2_path_planner`)
```

## Quick start

```powershell
# Learned planner (training needs CUDA; see NN/README.md for the full pipeline)
cd NN
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider

# Model-based planner
cd "HFSM_ Frenet_quintic_ LTV-MPC"
python -m pip install -e ".[test]"
python -m l2_path_planner run --config configs/baseline.json
pytest -q
```

## Not in the repository

Ignored on purpose (see `.gitignore`); each item can be rebuilt or re-cloned locally.

| Path | Size | How to restore |
| --- | --- | --- |
| `NN/data/` | ~22 GB | Download comma2k19, then `build_dataset.py` / `prepare_lanes.py` / `prepare_teacher.py` |
| `NN/models/` | ~825 MB | Polar R-CNN CULane checkpoint + comma2k19 fine-tune; SHA-256 hashes are in `NN/README.md` |
| `NN/third_party/PolarRCNN` | 12 MB | `git clone https://github.com/ShqWW/PolarRCNN.git` (pinned `7cc0c09`) |
| `NN/third_party/Ultra-Fast-Lane-Detection-v2` | 49 MB | `git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git` (pinned `c903880`) |
| `HFSM_ Frenet_quintic_ LTV-MPC/ui/data/` | 61 MB | `python -m l2_path_planner ui --all` |
| `HFSM_ Frenet_quintic_ LTV-MPC/outputs/` | 18 MB | `python -m l2_path_planner run --config configs/baseline.json` |

## Conventions

- Python ≥ 3.10, Windows/Linux, LF line endings, no committed datasets or weights.
- Reported metrics always come from route-level splits and recorded scenario runs;
  the bounded safety projection in `NN/safety_shield.py` is a diagnostic and never
  counts towards model acceptance.
