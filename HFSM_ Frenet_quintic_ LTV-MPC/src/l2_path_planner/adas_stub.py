"""ADAS / driver timeline (benchmark stimulus, not a planner input)."""

from __future__ import annotations

from dataclasses import dataclass

from .messages import Side


def _side(value) -> Side:
    text = str(value or "NONE").upper()
    return Side.LEFT if text == "LEFT" else Side.RIGHT if text == "RIGHT" else Side.NONE


@dataclass(slots=True)
class AdasState:
    hwa: bool = True
    lka: bool = True
    elk: bool = True
    acc_active: bool = True
    esc_enabled: bool = True
    esc_intervening: bool = False
    hands_on: bool = True
    turn_signal: Side = Side.NONE
    lane_change_request: Side = Side.NONE
    driver_torque_nm: float = 0.0
    driver_rate_dps: float = 0.0
    driver_lane_change_completed: bool = False
    desired_speed_mps: float = 20.0


class AdasTimeline:
    """Keyframe timeline: each entry overrides the fields it names."""

    def __init__(self, entries: list[dict], defaults: dict | None = None) -> None:
        self.entries = sorted((dict(entry) for entry in (entries or [])), key=lambda item: float(item.get("t", 0.0)))
        base = {
            "hwa": True, "lka": True, "elk": True, "acc_active": True,
            "esc_enabled": True, "esc_intervening": False, "hands_on": True,
            "turn_signal": "NONE", "lane_change_request": "NONE",
            "driver_torque_nm": 0.0, "driver_rate_dps": 0.0,
            "driver_lane_change_completed": False,
            "desired_speed_mps": 20.0,
        }
        base.update(defaults or {})
        self.base = base

    def at(self, t: float) -> AdasState:
        values = dict(self.base)
        for entry in self.entries:
            if float(entry.get("t", 0.0)) <= t + 1e-9:
                for key, value in entry.items():
                    if key != "t":
                        values[key] = value
        return AdasState(
            hwa=bool(values["hwa"]),
            lka=bool(values["lka"]),
            elk=bool(values["elk"]),
            acc_active=bool(values["acc_active"]),
            esc_enabled=bool(values["esc_enabled"]),
            esc_intervening=bool(values["esc_intervening"]),
            hands_on=bool(values["hands_on"]),
            turn_signal=_side(values.get("turn_signal")),
            lane_change_request=_side(values.get("lane_change_request")),
            driver_torque_nm=float(values.get("driver_torque_nm", 0.0)),
            driver_rate_dps=float(values.get("driver_rate_dps", 0.0)),
            driver_lane_change_completed=bool(values.get("driver_lane_change_completed", False)),
            desired_speed_mps=float(values.get("desired_speed_mps", self.base["desired_speed_mps"])),
        )
