"""Coordinate transforms shared by all dataset adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract planar yaw from a quaternion."""

    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)


@dataclass(frozen=True)
class LocalFrame:
    """Initial ego frame: +X forward, +Y left, +Z up."""

    origin_x: float
    origin_y: float
    origin_z: float
    origin_yaw: float

    def point(self, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
        dx = x - self.origin_x
        dy = y - self.origin_y
        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)
        return c * dx + s * dy, -s * dx + c * dy, z - self.origin_z

    def vector(self, x: float, y: float) -> tuple[float, float]:
        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)
        return c * x + s * y, -s * x + c * y

    def yaw(self, yaw: float) -> float:
        return normalize_angle(yaw - self.origin_yaw)
