# Dataset plan for ego-centric lateral planning

## Selected primary dataset: comma2k19

Official source: https://github.com/commaai/comma2k19

Reasons for selection:

- More than 33 hours of highway driving (2019 one-minute segments).
- Road-facing camera recorded at 20 Hz, exactly matching the TCN rate.
- IMU gyro and acceleration at 100 Hz.
- Processed CAN includes car speed, steering angle, wheel speeds and radar.
- Per-frame camera position and orientation are available for constructing a
  future ego trajectory label without an external viewpoint.
- Approximately 100 GB total, split into roughly 10 GB chunks.

Important limitation: comma2k19 was collected with openpilot logging. Before
treating a segment as a human demonstration, inspect the raw log engagement
state and remove automated-lateral-control periods. Do not infer manual driving
from steering angle alone.

## Human-driving reference dataset: DDD17

Official paper and download link: https://arxiv.org/abs/1711.01458

- More than 12 hours and 1000 km of real driving.
- Highway and city, day/night, dry/wet conditions.
- Conventional grayscale frames plus event-camera data.
- Explicit driver steering, throttle, brake, vehicle speed and GPS.

Use DDD17 to verify that the learned temporal behavior is also present in a
dataset explicitly described as driver-control data. Its camera resolution and
sensor type differ from the intended production camera, so it should not be the
only training source.

## Supplementary datasets

### A2D2

Official source: https://a2d2-dataset.github.io/

Use for perception pretraining/domain variation. It has synchronized cameras,
LiDAR and vehicle-bus data, but it is heavier and less directly aligned with a
20 Hz highway lateral-planning experiment.

### nuScenes CAN bus expansion

Official source:
https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/can_bus

Useful for cross-checking signal processing: IMU at 100 Hz, pose at 50 Hz and
steering feedback at 100 Hz. Scenes are short and urban, and the CAN expansion
is officially described as experimental, so it is not the primary dataset.

## Canonical 20 Hz training record

Each row represents one synchronized timestep:

```text
timestamp
lane_left_c0, lane_left_c1, lane_left_c2, lane_left_c3
lane_right_c0, lane_right_c1, lane_right_c2, lane_right_c3
lane_left_confidence, lane_right_confidence
vehicle_speed_mps
yaw_rate_rps
lateral_acceleration_mps2
steering_angle_rad
steering_rate_rps
validity_mask
manual_lateral_control
```

TCN input window:

```text
40 past/current samples x 16 model features = 2 seconds at 20 Hz
```

Recommended target:

```text
y_T_m
y_dot_T_mps
y_ddot_T_mps2
log_sigma_y (learned by Gaussian NLL; it is not a direct dataset label)
```

The target ego trajectory should be constructed from future per-frame poses in
the coordinate frame of the current camera. Global position is used only
offline to generate labels; it is not a runtime model input.

## Processing pipeline

1. Decode video and preserve the original frame timestamps.
2. Resample CAN and IMU signals to camera timestamps at 20 Hz.
3. Interpolate continuous signals; use nearest/previous sampling for discrete
   validity and engagement flags.
4. Run one fixed lane-perception model on the front-camera frames.
5. Fit left/right lane boundaries in the ego frame and store confidence.
6. Construct 2-3 second future ego trajectories from the supplied poses.
7. Fit the desired terminal lateral state or local quintic target.
8. Reject windows with invalid synchronization, implausible pose jumps or
   automated lateral control when training a human-driving policy.
9. Split train/validation/test by complete route or recording day, never by
   randomly shuffling overlapping windows.

## Synthetic augmentation calibrated from real logs

Generate synthetic variants only after measuring the real signal statistics:

- Lane coefficient noise and bias as a function of speed and curvature.
- One-frame and multi-frame c0 jumps.
- Loss of one boundary and loss of both boundaries.
- Perception delay of 50-200 ms.
- Confidence decay before a lane loss.
- IMU white noise, slowly varying bias and timestamp jitter.
- Steering-angle quantization and steering-rate limiting.
- Initial lateral and heading perturbations for LKA recovery.

ELK departures must be synthesized with a constrained expert planner or an
NCAP/OpenSCENARIO test definition. Normal human-driving logs are not sufficient
to supply safe emergency-departure demonstrations.

## Download strategy

1. Begin with the one-minute comma2k19 example stored in its GitHub repository.
2. Validate decoding, synchronization, lane extraction and label generation.
3. Download one roughly 10 GB chunk and train the first TCN baseline.
4. Download additional chunks only after the pipeline passes data-quality and
   leakage checks.

Do not download the full approximately 100 GB dataset before the one-minute and
single-chunk pipelines are verified.
