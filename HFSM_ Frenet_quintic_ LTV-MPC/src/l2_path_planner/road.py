"""Road model in the signal-list frame.

The whole code base uses the frame published by ``Signal List.xlsx``:

* ``x`` points forward along the road, ``y`` points to the **right**;
* a positive ``d`` (Frenet offset), ``c0`` (line offset), lateral velocity,
  yaw rate and steering angle all mean "to the right".

``curve_amplitude_m`` is therefore the **signed** lateral apex of the S-curve in
that frame: a positive value bends the road to the right, a negative value to
the left.  Keeping the sign in the scenario (rather than hard-coding a side)
lets the same physical scenario be described after the frame was mirrored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(slots=True)
class SimpleRoad:
    lane_width_m: float
    curve_start_m: float
    curve_length_m: float
    #: Signed lateral apex in the sheet frame (positive = to the right).
    curve_amplitude_m: float

    def _u(self, s: np.ndarray) -> np.ndarray:
        return (s - self.curve_start_m) / self.curve_length_m

    def geometry(self, s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_arr = np.asarray(s, dtype=float)
        u = self._u(s_arr)
        inside = (u >= 0.0) & (u <= 1.0)
        y = np.zeros_like(s_arr)
        dy = np.zeros_like(s_arr)
        ddy = np.zeros_like(s_arr)
        w = 2.0 * np.pi / self.curve_length_m
        phase = 2.0 * np.pi * u[inside]
        y[inside] = 0.5 * self.curve_amplitude_m * (1.0 - np.cos(phase))
        dy[inside] = 0.5 * self.curve_amplitude_m * w * np.sin(phase)
        ddy[inside] = 0.5 * self.curve_amplitude_m * w * w * np.cos(phase)
        yaw = np.arctan(dy)
        curvature = ddy / np.power(1.0 + dy * dy, 1.5)
        return s_arr, y, yaw, curvature

    def project(self, x_m: float, y_m: float) -> tuple[float, float, float, float]:
        s = float(x_m)
        _, yc, yaw, curvature = self.geometry(np.array([s]))
        d = (y_m - float(yc[0])) * math.cos(float(yaw[0]))
        return s, d, float(yaw[0]), float(curvature[0])

    def frenet_to_global(
        self, s: np.ndarray, d: np.ndarray, d_dot: np.ndarray, speed_mps: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_arr, yc, yaw_c, kappa_c = self.geometry(s)
        x = s_arr - d * np.sin(yaw_c)
        y = yc + d * np.cos(yaw_c)
        yaw = wrap_angle(yaw_c + np.arctan2(d_dot, max(speed_mps, 0.1)))
        return x, y, np.asarray(yaw), kappa_c

