"""Time-series resampling for canonical 10 Hz trajectories."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Optional

from .geometry import normalize_angle


TARGET_DT = 0.1


@dataclass
class Sample:
    timestamp: float
    x: float
    y: float
    yaw: float
    vx: float
    vy: float
    valid: bool = True


def build_timeline(start: float, end: float, dt: float = TARGET_DT) -> list[float]:
    """Build an inclusive, numerically stable fixed-rate timeline."""

    if end < start:
        raise ValueError("timeline end precedes start")
    count = int(math.floor((end - start) / dt + 1e-8)) + 1
    return [round(start + index * dt, 9) for index in range(count)]


def _lerp(left: float, right: float, ratio: float) -> float:
    return left + ratio * (right - left)


def _lerp_yaw(left: float, right: float, ratio: float) -> float:
    return normalize_angle(left + ratio * normalize_angle(right - left))


def resample_samples(
    samples: list[Sample],
    timeline: list[float],
    max_gap: Optional[float] = None,
) -> dict[str, list[object]]:
    """Interpolate valid adjacent samples without bridging missing intervals."""

    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    times = [sample.timestamp for sample in ordered]
    result: dict[str, list[object]] = {
        "x": [],
        "y": [],
        "yaw": [],
        "vx": [],
        "vy": [],
        "valid": [],
    }

    for target in timeline:
        right_index = bisect.bisect_left(times, target)
        if right_index < len(times) and math.isclose(times[right_index], target, abs_tol=1e-7):
            left_index = right_index
        else:
            left_index = right_index - 1

        valid = 0 <= left_index < len(ordered) and 0 <= right_index < len(ordered)
        if valid:
            left = ordered[left_index]
            right = ordered[right_index]
            gap = right.timestamp - left.timestamp
            valid = left.valid and right.valid and (max_gap is None or gap <= max_gap + 1e-7)
        if not valid:
            for key in ("x", "y", "yaw", "vx", "vy"):
                result[key].append(None)
            result["valid"].append(False)
            continue

        if left_index == right_index:
            ratio = 0.0
        else:
            ratio = (target - left.timestamp) / (right.timestamp - left.timestamp)
        result["x"].append(_lerp(left.x, right.x, ratio))
        result["y"].append(_lerp(left.y, right.y, ratio))
        result["yaw"].append(_lerp_yaw(left.yaw, right.yaw, ratio))
        result["vx"].append(_lerp(left.vx, right.vx, ratio))
        result["vy"].append(_lerp(left.vy, right.vy, ratio))
        result["valid"].append(True)

    return result


def resample_discrete_states(
    samples: list[tuple[float, str]],
    timeline: list[float],
) -> tuple[list[str], list[bool], list[Optional[str]]]:
    """Resample traffic lights with previous-value hold."""

    ordered = sorted(samples)
    times = [sample[0] for sample in ordered]
    states: list[str] = []
    valid: list[bool] = []
    source_states: list[Optional[str]] = []
    for target in timeline:
        index = bisect.bisect_right(times, target) - 1
        if index < 0:
            states.append("unknown")
            valid.append(False)
            source_states.append(None)
        else:
            source = ordered[index][1]
            states.append(source)
            valid.append(True)
            source_states.append(source)
    return states, valid, source_states
