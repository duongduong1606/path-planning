"""Lateral state estimation from the signals the vehicle actually publishes.

The signal list has no lateral-velocity channel, so ``v_y`` must be
reconstructed.  This module keeps a two-state Kalman filter ``[d, v_y]``:

* prediction runs at the IMU rate and uses the body-frame lateral acceleration
  (``v_dot_y = a_y - v * r``) plus the lane heading;
* correction runs whenever a new camera frame arrives and uses the offset
  derived from the two published line offsets.

Both ``d`` and ``v_y`` live in the same lateral sign convention as the lane
offsets, so the filter is unchanged by the frame convention itself.
"""

from __future__ import annotations

import numpy as np

from .messages import LaneModel


class LateralStateEstimator:
    """Two-state Kalman filter for ``[lateral_offset_m, lateral_velocity_mps]``."""

    def __init__(
        self,
        dt_s: float = 0.01,
        measurement_std_m: float = 0.02,
        offset_process_std_m: float = 0.01,
        velocity_process_std_mps: float = 0.5,
        initial_velocity_std_mps: float = 0.5,
    ) -> None:
        self.dt = float(dt_s)
        self.measurement_variance = float(measurement_std_m) ** 2
        self.q = np.diag([
            float(offset_process_std_m) ** 2,
            float(velocity_process_std_mps) ** 2,
        ])
        self.initial_velocity_variance = float(initial_velocity_std_mps) ** 2
        self.reset()

    def reset(self) -> None:
        self._x = np.zeros(2)
        self._P = np.diag([1.0, self.initial_velocity_variance])
        self._initialised = False
        self._last_stamp_us: int | None = None
        self._camera_age_s = float("inf")

    # -- inputs ------------------------------------------------------------
    @staticmethod
    def _lane_offset_and_heading(lane: LaneModel) -> tuple[float, float] | None:
        """Offset of the ego from the lane centre and the lane heading error.

        ``c0`` is positive to the right; the ego sits at ``-mean(c0)`` relative
        to the lane centre.  ``c1`` is the lane heading relative to the ego, so
        the heading error of the ego relative to the lane is ``-c1``.
        """
        left, right = lane.left, lane.right
        if left is not None and right is not None:
            offset = -0.5 * (left.c0 + right.c0)
            return offset, -left.c1
        line = left if left is not None else right
        if line is not None:
            # Single line: keep the remembered half width out of the offset and
            # only use the heading, so the position is not corrupted.
            return float("nan"), -line.c1
        return None

    # -- filter ------------------------------------------------------------
    def update(
        self,
        lane: LaneModel | None,
        speed_mps: float,
        yaw_rate_rps: float,
        lateral_acceleration_mps2: float,
    ) -> None:
        speed = float(speed_mps)
        heading = 0.0
        measurement: float | None = None

        if lane is not None:
            derived = self._lane_offset_and_heading(lane)
            if derived is not None:
                offset, heading = derived
                if lane.header.stamp_us != self._last_stamp_us:
                    self._last_stamp_us = lane.header.stamp_us
                    self._camera_age_s = 0.0
                    if np.isfinite(offset):
                        measurement = float(offset)
                else:
                    self._camera_age_s += self.dt

        # Predict: d_dot = v_y + v * heading_error, v_dot_y = a_y - v * r
        f = np.array([[1.0, self.dt], [0.0, 1.0]])
        drive = np.array([
            self.dt * speed * heading,
            self.dt * (float(lateral_acceleration_mps2) - speed * float(yaw_rate_rps)),
        ])
        self._x = f @ self._x + drive
        self._P = f @ self._P @ f.T + self.q * self.dt

        if measurement is None:
            return

        if not self._initialised:
            self._x[0] = measurement
            self._P[0, 0] = self.measurement_variance
            self._initialised = True

        h = np.array([[1.0, 0.0]])
        predicted = float((h @ self._x)[0])
        innovation = measurement - predicted
        s = float((h @ self._P @ h.T)[0, 0]) + self.measurement_variance
        gain = (self._P @ h.T) / s
        self._x = self._x + (gain * innovation).reshape(-1)
        self._P = (np.eye(2) - gain @ h) @ self._P

    # -- outputs -----------------------------------------------------------
    @property
    def offset_m(self) -> float:
        return float(self._x[0])

    @property
    def lateral_velocity_mps(self) -> float:
        return float(self._x[1])

    @property
    def variance(self) -> float:
        return float(self._P[1, 1])

    @property
    def camera_age_s(self) -> float:
        return self._camera_age_s

    @property
    def initialised(self) -> bool:
        return self._initialised
