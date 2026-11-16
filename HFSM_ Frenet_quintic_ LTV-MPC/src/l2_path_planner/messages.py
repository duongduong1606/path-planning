from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class Function(IntEnum):
    NONE = 0
    HWA = 1
    LKA = 2
    ELK = 3


class RequestState(IntEnum):
    OFF = 0
    UNAVAILABLE = 1
    STANDBY = 2
    REQUEST = 3
    FAULT = 4


class Side(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


class InterventionPhase(IntEnum):
    NONE = 0
    ACTIVE = 1
    COMPLETED = 2


class LineStatus(IntEnum):
    MEASURED = 0
    PREDICTED = 1
    LOST = 2


class LateralManeuver(IntEnum):
    NONE = 0
    LANE_CENTER = 1
    OFFSET_LEFT = 2
    OFFSET_RIGHT = 3
    LANE_CHANGE_LEFT = 4
    LANE_CHANGE_RIGHT = 5
    RECOVERY_LEFT = 6
    RECOVERY_RIGHT = 7


class LongitudinalMode(IntEnum):
    HOLD = 0
    CRUISE = 1
    ACCELERATE = 2
    FOLLOW = 3
    BRAKE = 4
    EMERGENCY_BRAKE = 5


@dataclass(slots=True)
class Header:
    stamp_us: int
    counter: int
    valid: bool = True
    frame_id: str = "base_link"


#: Line classes that must not be crossed (used to derive ``crossable``).
SOLID_LINE_CLASSES = frozenset({"SOLID", "DOUBLE", "ROAD_EDGE", "BARRIER"})

#: Minimum / maximum plausible lane width, used when deriving it from two lines.
LANE_WIDTH_MIN_M = 2.2
LANE_WIDTH_MAX_M = 5.5


@dataclass(slots=True)
class LaneLine:
    """One line as published by the camera.

    Field names and units follow the signal list: ``c0`` lateral offset (m),
    ``c1`` heading (rad), ``c2_inv_km`` curvature coefficient (1/km),
    ``view_range_*_m`` start/end distance (m), ``line_class``, and
    ``existence_probability`` (ratio).

    ``c3`` is kept for the interface but is structurally zero: the camera model
    is quadratic.  ``crossable``, ``confidence`` and ``line_type`` are derived,
    not published.  ``variance`` and ``status`` have no signal in the sheet and
    are carried unchanged for interface compatibility (case 3: unresolved).
    """

    c0: float
    c1: float
    c2_inv_km: float = 0.0
    c3: float = 0.0
    view_range_start_m: float = 0.0
    view_range_end_m: float = 100.0
    line_class: str = "DASHED"
    existence_probability: float = 1.0
    #: Not published by the sheet; left untouched (case 3).
    variance: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: Not published by the sheet; left untouched (case 3).
    status: LineStatus = LineStatus.MEASURED

    @property
    def c2_inv_m(self) -> float:
        return self.c2_inv_km / 1000.0

    @property
    def curvature_inv_m(self) -> float:
        return 2.0 * self.c2_inv_m

    @property
    def crossable(self) -> bool:
        return self.line_class.upper() not in SOLID_LINE_CLASSES

    @property
    def line_type(self) -> str:
        return self.line_class

    @property
    def confidence(self) -> float:
        return self.existence_probability

    @property
    def view_range_end(self) -> float:
        return self.view_range_end_m


@dataclass(slots=True)
class LaneModel:
    header: Header
    left: LaneLine | None
    right: LaneLine | None
    lane_width: float = 3.5
    next_left: LaneLine | None = None
    next_right: LaneLine | None = None

    def derived_lane_width(self) -> float:
        """Lane width from the two inner lines; ``lane_width`` is the memory."""
        if self.left is not None and self.right is not None:
            width = abs(self.left.c0 - self.right.c0)
            if LANE_WIDTH_MIN_M <= width <= LANE_WIDTH_MAX_M:
                return width
        return self.lane_width


@dataclass(slots=True)
class FunctionRequest:
    function: Function
    state: RequestState
    side: Side = Side.NONE
    tlc_s: float = 99.0
    reason: str = ""
    target_offset_m: float = 0.0
    maneuver: LateralManeuver = LateralManeuver.NONE
    elk_reason: str = "NONE"
    cancel_reason: str = "NONE"


@dataclass(slots=True)
class FunctionRequests:
    header: Header
    requests: list[FunctionRequest]
    driver_override: bool = False

    def request_for(self, function: Function) -> FunctionRequest | None:
        return next((r for r in self.requests if r.function == function), None)


@dataclass(slots=True)
class VehicleStateEstimate:
    header: Header
    lateral_velocity_mps: float
    lateral_velocity_variance: float
    slip_angle_rad: float


@dataclass(slots=True)
class EgoMotion:
    valid: bool
    dx_m: float
    dy_m: float
    dyaw_rad: float


@dataclass(slots=True)
class TrajectoryPoint:
    time_from_start_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    longitudinal_velocity_mps: float
    lateral_velocity_mps: float
    acceleration_mps2: float
    heading_rate_rps: float
    curvature_inv_m: float
    front_wheel_angle_rad: float
    rear_wheel_angle_rad: float = 0.0
    road_s_m: float = 0.0
    road_d_m: float = 0.0


@dataclass(slots=True)
class Trajectory:
    header: Header
    points: list[TrajectoryPoint]


@dataclass(slots=True)
class PathStatus:
    header: Header
    path_valid: bool
    active_function: Function
    intervention_phase: InterventionPhase
    lateral_limit_reached: bool
    compute_time_ms: float


@dataclass(slots=True)
class LateralLimits:
    header: Header
    v_max_mps: float
    distance_m: float
    lateral_limit_reached: bool


@dataclass(slots=True)
class LongitudinalPlan:
    header: Header
    mode: LongitudinalMode
    target_speed_mps: float
    acceleration_mps2: float
    lead_gap_m: float = 1e6
    reason: str = ""


def active_request(functions: Iterable[FunctionRequest], function: Function) -> bool:
    return any(r.function == function and r.state == RequestState.REQUEST for r in functions)

