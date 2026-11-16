"""Signal-list boundary layer.

Source of truth: ``Signal List.xlsx`` (groups Motion / Perception / Radar).
This module is the only place where

* units are converted to SI (the sheet publishes km/h, deg/s, deg, 1/km, %),
* derived channels are reconstructed from the channels that really exist,
* the signal rates declared by the sheet are kept in one table.

Nothing here invents a signal the sheet does not contain.  Signals that are
absent are listed in :data:`ABSENT_SIGNALS` and must come from outside this
layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .messages import Header

# --------------------------------------------------------------------------
# Rates declared by the sheet (Hz)
# --------------------------------------------------------------------------
RATES_HZ: dict[str, float] = {
    # Motion / ego
    "vehicle_speed": 50.0,
    "wheel_speed": 50.0,
    "wheel_angular_speed": 50.0,
    "wheel_counter": 50.0,
    "wheel_direction": 50.0,
    "standstill": 50.0,
    "road_slope": 50.0,
    "driver_steering_torque": 50.0,
    "steering_motor_torque": 50.0,
    "accelerator_pedal": 100.0,
    "brake_pedal": 50.0,
    "brake_pedal_travel": 50.0,
    "front_motor_speed": 100.0,
    "front_motor_torque": 100.0,
    "friction_brake_torque": 25.0,
    "master_cylinder_pressure": 50.0,
    "brake_pressure": 50.0,
    "odometer": 50.0,
    "longitudinal_acceleration": 100.0,
    "lateral_acceleration": 100.0,
    "yaw_rate": 100.0,
    "steering_wheel_angle": 100.0,
    "steering_wheel_speed": 100.0,
    "steering_torque": 100.0,
    # Perception / Radar / ground truth
    "camera": 10.0,
    "radar": 20.0,
    "ground_truth": 100.0,
}

#: Signals the planner needs but the sheet does not publish.  Kept explicit so
#: that no code silently substitutes ground truth for a missing signal.
ABSENT_SIGNALS: dict[str, str] = {
    "lateral_velocity": "Motion has no v_y; G2 VehicleStateEstimate is not in the sheet",
    "lane_width": "no lane-width field; derived from left/right line offsets",
    "lane_change": "no lane-change flag; must be inferred or supplied by ADAS",
    "road_bank": "only Road slope (%) exists, which is longitudinal grade",
    "friction_coefficient": "not published",
    "object_velocity": "camera objects carry no vx/vy; only rear radars publish relative speed",
    "object_size": "camera objects carry class only; size comes from a class table",
    "turn_signal": "not published",
    "timestamp": "no clock/stamp columns; Header stamps stay internal",
}

# --------------------------------------------------------------------------
# Unit conversions
# --------------------------------------------------------------------------
MPS_TO_KMH = 3.6


def kmh_to_mps(value_kmh: float) -> float:
    return value_kmh / MPS_TO_KMH


def mps_to_kmh(value_mps: float) -> float:
    return value_mps * MPS_TO_KMH


def deg_to_rad(value_deg: float) -> float:
    return math.radians(value_deg)


def rad_to_deg(value_rad: float) -> float:
    return math.degrees(value_rad)


def degps_to_radps(value_degps: float) -> float:
    return math.radians(value_degps)


def radps_to_degps(value_radps: float) -> float:
    return math.degrees(value_radps)


def curvature_inv_km_to_inv_m(value_inv_km: float) -> float:
    return value_inv_km / 1000.0


def curvature_inv_m_to_inv_km(value_inv_m: float) -> float:
    return value_inv_m * 1000.0


def steering_wheel_deg_to_wheel_rad(value_deg: float, steering_ratio: float) -> float:
    return math.radians(value_deg) / steering_ratio


def wheel_rad_to_steering_wheel_deg(value_rad: float, steering_ratio: float) -> float:
    return math.degrees(value_rad * steering_ratio)


def config_speed_mps(vehicle_cfg: dict) -> float:
    """Vehicle speed in m/s.

    The sheet publishes km/h, so ``speed_kmh`` wins when present; the legacy
    ``speed_mps`` key is accepted for configs written in SI.
    """
    if "speed_kmh" in vehicle_cfg:
        return kmh_to_mps(float(vehicle_cfg["speed_kmh"]))
    return float(vehicle_cfg["speed_mps"])


# --------------------------------------------------------------------------
# Derived-channel helpers
# --------------------------------------------------------------------------
class LowPass:
    """First-order low pass used to build the `filtered` variants."""

    def __init__(self, tau_s: float, dt_s: float) -> None:
        self.tau_s = max(float(tau_s), 0.0)
        self.dt_s = float(dt_s)
        self._value: float | None = None

    def __call__(self, sample: float) -> float:
        if self._value is None or self.tau_s <= 0.0:
            self._value = float(sample)
        else:
            alpha = self.dt_s / (self.tau_s + self.dt_s)
            self._value += alpha * (float(sample) - self._value)
        return self._value

    def reset(self) -> None:
        self._value = None


# --------------------------------------------------------------------------
# Motion group
# --------------------------------------------------------------------------
@dataclass(slots=True)
class ChassisSignals:
    """Motion signals in SI units, plus the sheet-unit views as properties."""

    vehicle_speed_mps: float
    yaw_rate_raw_radps: float
    yaw_rate_filtered_radps: float
    ay_raw_mps2: float
    ay_filtered_mps2: float
    ax_raw_mps2: float
    ax_filtered_mps2: float
    steering_wheel_angle_rad: float
    steering_wheel_rate_radps: float
    steering_torque_nm: float
    driver_steering_torque_nm: float
    standstill: bool
    road_slope_percent: float
    wheel_speed_mps: tuple[float, float, float, float]
    wheel_angular_speed_radps: tuple[float, float, float, float]
    wheel_counter: tuple[int, int, int, int] = (0, 0, 0, 0)
    odometer_m: float = 0.0
    front_motor_speed_rpm: float = 0.0
    front_motor_torque_nm: float = 0.0

    @property
    def vehicle_speed_kmh(self) -> float:
        return mps_to_kmh(self.vehicle_speed_mps)

    @property
    def yaw_rate_raw_degps(self) -> float:
        return radps_to_degps(self.yaw_rate_raw_radps)

    @property
    def yaw_rate_filtered_degps(self) -> float:
        return radps_to_degps(self.yaw_rate_filtered_radps)

    @property
    def steering_wheel_angle_deg(self) -> float:
        return rad_to_deg(self.steering_wheel_angle_rad)

    @property
    def steering_wheel_rate_degps(self) -> float:
        return radps_to_degps(self.steering_wheel_rate_radps)


class ChassisSignalModel:
    """Builds :class:`ChassisSignals` from the plant state.

    Raw channels are the plant values; filtered channels are a first-order low
    pass of the raw channel, standing in for the ECU filter named in the sheet
    ("Yaw rate, filtered", "Lateral acceleration, filtered").
    """

    def __init__(self, vehicle_cfg: dict, dt_s: float, filter_tau_s: float = 0.05,
                 road_slope_percent: float = 0.0) -> None:
        self.vehicle = vehicle_cfg
        self.dt = float(dt_s)
        self.steering_ratio = float(vehicle_cfg["steering_ratio"])
        self.wheel_radius_m = float(vehicle_cfg.get("wheel_radius_m", 0.34))
        self.road_slope_percent = float(road_slope_percent)
        self._yaw = LowPass(filter_tau_s, dt_s)
        self._ay = LowPass(filter_tau_s, dt_s)
        self._ax = LowPass(filter_tau_s, dt_s)
        self._previous_speed: float | None = None
        self._previous_steering: float | None = None
        self._odometer_m = 0.0
        self._wheel_angle_rad = 0.0

    def update(self, plant_state, speed_mps: float, driver_torque_nm: float = 0.0) -> ChassisSignals:
        speed = float(speed_mps)
        yaw_rate = float(plant_state.yaw_rate_rps)
        # Body-frame lateral acceleration at the CG (IMU channel).  This is not
        # the centripetal v*r term, so `a_y - v*r` recovers v_dot_y.
        ay = float(getattr(plant_state, "lateral_acceleration_mps2", speed * yaw_rate))

        if self._previous_speed is None:
            ax = 0.0
        else:
            ax = (speed - self._previous_speed) / self.dt
        self._previous_speed = speed

        steering = float(plant_state.steering_angle_rad)
        if self._previous_steering is None:
            steering_rate = 0.0
        else:
            steering_rate = (steering - self._previous_steering) / self.dt
        self._previous_steering = steering

        self._odometer_m += speed * self.dt
        self._wheel_angle_rad += speed * self.dt / max(self.wheel_radius_m, 1e-6)

        wheel_speed = (speed, speed, speed, speed)
        wheel_angular = tuple(value / max(self.wheel_radius_m, 1e-6) for value in wheel_speed)
        count = int(self._wheel_angle_rad / (2.0 * math.pi) * 1000.0) % 65536
        wheel_counter = (count, count, count, count)

        return ChassisSignals(
            vehicle_speed_mps=speed,
            yaw_rate_raw_radps=yaw_rate,
            yaw_rate_filtered_radps=self._yaw(yaw_rate),
            ay_raw_mps2=ay,
            ay_filtered_mps2=self._ay(ay),
            ax_raw_mps2=ax,
            ax_filtered_mps2=self._ax(ax),
            steering_wheel_angle_rad=steering * self.steering_ratio,
            steering_wheel_rate_radps=steering_rate * self.steering_ratio,
            steering_torque_nm=0.0,
            driver_steering_torque_nm=float(driver_torque_nm),
            standstill=bool(speed < 0.1),
            road_slope_percent=self.road_slope_percent,
            wheel_speed_mps=wheel_speed,
            wheel_angular_speed_radps=wheel_angular,
            wheel_counter=wheel_counter,
            odometer_m=self._odometer_m,
            front_motor_speed_rpm=speed / max(self.wheel_radius_m, 1e-6) * 60.0 / (2.0 * math.pi),
            front_motor_torque_nm=0.0,
        )


# --------------------------------------------------------------------------
# Perception / Radar groups
# --------------------------------------------------------------------------
@dataclass(slots=True)
class CameraObject:
    object_id: int
    object_class: str
    x_m: float
    y_m: float
    heading_rad: float
    existence_probability: float


@dataclass(slots=True)
class RadarObject:
    object_id: int
    x_m: float
    y_m: float
    vx_rel_mps: float
    vy_rel_mps: float
    rcs_dbsm: float
    motion_state: str
    existence_probability: float


@dataclass(slots=True)
class ObjectList:
    header: Header
    camera: list[CameraObject] = field(default_factory=list)
    radar_rear_left: list[RadarObject] = field(default_factory=list)
    radar_rear_right: list[RadarObject] = field(default_factory=list)


#: Class -> (length, width) in metres.  The sheet publishes class only.
OBJECT_CLASS_SIZE_M: dict[str, tuple[float, float]] = {
    "CAR": (4.5, 1.9),
    "TRUCK": (10.0, 2.5),
    "BUS": (12.0, 2.55),
    "MOTORCYCLE": (2.0, 0.8),
    "UNKNOWN": (4.5, 1.9),
}


class ObjectSignalModel:
    """Synthesises the camera and rear-radar views from the scenario stimuli."""

    CAMERA_RANGE_M = 80.0
    CAMERA_HALF_WIDTH_M = 5.25
    RADAR_RANGE_M = 60.0
    RADAR_MOUNT_X_M = -1.0
    RADAR_MOUNT_Y_M = 0.9

    def __init__(self, stimuli: dict, lane_width_m: float) -> None:
        self.stimuli = dict(stimuli)
        self.lane_width_m = float(lane_width_m)

    def _scenario_objects(self, t: float, ego_speed_mps: float) -> list[dict]:
        objects: list[dict] = []
        truck_start = float(self.stimuli.get("truck_start_s", 0.0))
        truck_end = float(self.stimuli.get("truck_end_s", -1.0))
        if "truck_start_s" in self.stimuli and truck_start <= t < truck_end:
            side = str(self.stimuli.get("truck_side", "LEFT")).upper()
            # Sheet frame: the left adjacent lane has a negative lateral offset.
            default_y = -self.lane_width_m if side != "RIGHT" else self.lane_width_m
            objects.append({
                "id": 1,
                "class": "TRUCK",
                "x": float(self.stimuli.get("truck_x_m", 20.0)),
                "y": float(self.stimuli.get("truck_y_m", default_y)),
                "vx": ego_speed_mps - float(self.stimuli.get("truck_delta_speed_mps", 5.0)),
            })
        lead_start = float(self.stimuli.get("lead_start_s", 0.0))
        lead_end = float(self.stimuli.get("lead_end_s", -1.0))
        if "lead_start_s" in self.stimuli and lead_start <= t < lead_end:
            objects.append({
                "id": 2,
                "class": "CAR",
                "x": float(self.stimuli.get("lead_gap_m", 30.0)),
                "y": 0.0,
                "vx": float(self.stimuli.get("lead_speed_mps", 17.0)),
            })
        # A rear approach is described by its TTC; the rear radars see it as a
        # closing object behind the ego vehicle.
        rear = self.stimuli.get("rear_vehicle") or {}
        if rear and float(rear.get("start_s", 0.0)) <= t < float(rear.get("end_s", 0.0)):
            gap = float(rear.get("gap_m", 10.0))
            ttc = max(float(rear.get("ttc_s", 2.0)), 1e-3)
            closing = gap / ttc
            side = str(rear.get("side", "LEFT")).upper()
            objects.append({
                "id": 3,
                "class": str(rear.get("class", "CAR")),
                "x": -gap,
                "y": -self.lane_width_m if side != "RIGHT" else self.lane_width_m,
                "vx": ego_speed_mps - closing,
            })
        return objects

    def camera_objects(self, t: float, ego_speed_mps: float, header: Header) -> list[CameraObject]:
        """Camera view, sampled at the camera rate declared by the sheet."""
        camera: list[CameraObject] = []
        for obj in self._scenario_objects(t, ego_speed_mps):
            x, y = obj["x"], obj["y"]
            if -5.0 <= x <= self.CAMERA_RANGE_M and abs(y) <= self.CAMERA_HALF_WIDTH_M:
                camera.append(CameraObject(
                    object_id=obj["id"], object_class=obj["class"],
                    x_m=x, y_m=y, heading_rad=0.0, existence_probability=1.0,
                ))
        return camera

    def radar_objects(
        self, t: float, ego_speed_mps: float, header: Header,
    ) -> tuple[list[RadarObject], list[RadarObject]]:
        """Rear-left / rear-right radar views, sampled at the radar rate."""
        rear_left: list[RadarObject] = []
        rear_right: list[RadarObject] = []
        for obj in self._scenario_objects(t, ego_speed_mps):
            x, y = obj["x"], obj["y"]
            if -self.RADAR_RANGE_M <= x < 0.0:
                # Sheet frame: the left side of the vehicle is the negative side.
                left_side = y <= 0.0
                mount_y = -self.RADAR_MOUNT_Y_M if left_side else self.RADAR_MOUNT_Y_M
                radar = RadarObject(
                    object_id=obj["id"],
                    x_m=x - self.RADAR_MOUNT_X_M,
                    y_m=y - mount_y,
                    vx_rel_mps=obj["vx"] - ego_speed_mps,
                    vy_rel_mps=0.0,
                    rcs_dbsm=10.0 if obj["class"] == "TRUCK" else 0.0,
                    motion_state="MOVING",
                    existence_probability=1.0,
                )
                (rear_left if left_side else rear_right).append(radar)
        return rear_left, rear_right

    def update(self, t: float, ego_speed_mps: float, header: Header) -> ObjectList:
        rear_left, rear_right = self.radar_objects(t, ego_speed_mps, header)
        return ObjectList(header, self.camera_objects(t, ego_speed_mps, header), rear_left, rear_right)


def object_size_m(object_class: str) -> tuple[float, float]:
    return OBJECT_CLASS_SIZE_M.get(str(object_class).upper(), OBJECT_CLASS_SIZE_M["UNKNOWN"])
