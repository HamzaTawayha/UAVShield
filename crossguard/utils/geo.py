from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

from crossguard.defense.state import GeoPoint

EARTH_RADIUS_M = 6_371_000.0


def horizontal_distance_m(a: GeoPoint, b: GeoPoint) -> float:
    """Return haversine ground distance in meters."""

    lat1 = radians(a.lat)
    lat2 = radians(b.lat)
    dlat = radians(b.lat - a.lat)
    dlon = radians(b.lon - a.lon)

    h = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * atan2(sqrt(h), sqrt(1.0 - h))


def distance_3d_m(a: GeoPoint, b: GeoPoint) -> float:
    horizontal = horizontal_distance_m(a, b)
    vertical = b.alt_m - a.alt_m
    return sqrt(horizontal**2 + vertical**2)


def required_speed_mps(a: GeoPoint, b: GeoPoint, dt_s: float) -> float:
    if dt_s <= 0:
        return float("inf")
    return horizontal_distance_m(a, b) / dt_s


def wrap_angle_delta_deg(a: float, b: float) -> float:
    """Return shortest absolute difference between two angles in degrees."""

    delta = (b - a + 180.0) % 360.0 - 180.0
    return abs(delta)
