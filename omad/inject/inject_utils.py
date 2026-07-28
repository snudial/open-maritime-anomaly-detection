"""
OMAD - Injection Utilities
================================
Common utilities for A1/A2/A3 anomaly injection.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# =========================
# Constants
# =========================
EARTH_RADIUS = 6_371_000.0  # meters
KNOT_TO_MPS = 0.514444      # 1 knot = 0.514444 m/s
NM_TO_M = 1852.0            # 1 nautical mile = 1852 m

# Clipping thresholds (from statistics)
EPSILON_SED = 110.494501    # Q10 of SED_local (meters)
EPSILON_A = 0.0001          # Q10 of a_local (m/s²)
EPSILON_OMEGA = 3.0         # degrees

# Default injection parameters
THETA_G = 3.0               # SED multiplier for A1
THETA_V = 2.0               # Speed/heading multiplier for A2
ALPHA_STEP = 5.0            # meters, step size for alpha search


# =========================
# Data Classes
# =========================
@dataclass
class InjectionPlan:
    """Injection plan loaded from JSON."""
    route_id: str
    anomaly_type: str  # "A1" or "A2"
    T: int
    K: int
    scores: List[float]

    def get_top_k_indices(self) -> List[int]:
        """
        Return indices of top-K scores (scattered, not necessarily consecutive).
        Used by A1 injection where scattered anomalies are acceptable.
        """
        scores = np.array(self.scores)
        return list(np.argsort(scores)[-self.K:])


@dataclass
class A3Plan:
    """A3 injection plan."""
    route_id: str
    scores: List[float]


# =========================
# Score Utilities
# =========================
def minmax_normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize scores within a single route."""
    if not scores:
        return []
    arr = np.asarray(scores, dtype=float)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    v_range = v_max - v_min
    if v_range <= 0.0:
        return [0.0 for _ in scores]
    norm = (arr - v_min) / v_range
    return [float(x) for x in norm]


def build_track_index(
    input_csv: str,
) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], int]]:
    """
    Build track index from input CSV.
    Returns:
      - track_index: track_id -> list of rows sorted by timestamp
      - track_pos: (track_id, timestamp) -> index in track_index list
    Duplicates (same track_id, timestamp) are deduped.
    """
    track_points: Dict[str, Dict[str, dict]] = {}
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row.get("TRACK_ID")
            ts = row.get("TIMESTAMP")
            if track_id is None or ts is None:
                continue
            if track_id not in track_points:
                track_points[track_id] = {}
            if ts not in track_points[track_id]:
                track_points[track_id][ts] = row

    track_index: Dict[str, List[dict]] = {}
    track_pos: Dict[Tuple[str, str], int] = {}
    for track_id, ts_map in track_points.items():
        ts_sorted = sorted(ts_map.keys())
        rows = [ts_map[ts] for ts in ts_sorted]
        track_index[track_id] = rows
        for i, ts in enumerate(ts_sorted):
            track_pos[(track_id, ts)] = i

    return track_index, track_pos


# =========================
# Coordinate Utilities
# =========================
def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate distance between two points in meters using Haversine formula."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(EARTH_RADIUS * c)


def bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate bearing from point 1 to point 2 in degrees (0-360)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing_rad = np.arctan2(x, y)
    return float((np.degrees(bearing_rad) + 360) % 360)


def destination_point(lon: float, lat: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """Calculate destination point given start, bearing, and distance."""
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    brng = np.radians(bearing_deg)
    d = distance_m / EARTH_RADIUS

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brng))
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2)
    )

    return float(np.degrees(lon2)), float(np.degrees(lat2))


def cross_track_distance(
    lon_a: float, lat_a: float,
    lon_b: float, lat_b: float,
    lon_c: float, lat_c: float
) -> float:
    """
    Calculate perpendicular distance from point C to line AB (in meters).
    Uses cross-track distance formula for spherical geometry.
    """
    d_ac = haversine_distance(lon_a, lat_a, lon_c, lat_c)
    d_ab = haversine_distance(lon_a, lat_a, lon_b, lat_b)

    if d_ab < 1e-10:
        return d_ac

    lat_a_rad, lat_c_rad = np.radians(lat_a), np.radians(lat_c)
    lon_a_rad, lon_c_rad = np.radians(lon_a), np.radians(lon_c)
    lat_b_rad, lon_b_rad = np.radians(lat_b), np.radians(lon_b)

    # Bearing A to C
    y = np.sin(lon_c_rad - lon_a_rad) * np.cos(lat_c_rad)
    x = np.cos(lat_a_rad) * np.sin(lat_c_rad) - np.sin(lat_a_rad) * np.cos(lat_c_rad) * np.cos(lon_c_rad - lon_a_rad)
    bearing_ac = np.arctan2(y, x)

    # Bearing A to B
    y = np.sin(lon_b_rad - lon_a_rad) * np.cos(lat_b_rad)
    x = np.cos(lat_a_rad) * np.sin(lat_b_rad) - np.sin(lat_a_rad) * np.cos(lat_b_rad) * np.cos(lon_b_rad - lon_a_rad)
    bearing_ab = np.arctan2(y, x)

    # Cross-track distance
    cross_track = np.abs(
        np.arcsin(np.sin(d_ac / EARTH_RADIUS) * np.sin(bearing_ac - bearing_ab)) * EARTH_RADIUS
    )
    return float(cross_track)


def wrap_angle(angle: float) -> float:
    """Wrap angle to -180° ~ 180°."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def heading_from_sin_cos(sin_val: float, cos_val: float) -> float:
    """Calculate heading in degrees from sin and cos values."""
    return float(np.degrees(np.arctan2(sin_val, cos_val)))


# =========================
# SED Calculation (A1)
# =========================
def calculate_sed(
    lon_prev: float, lat_prev: float,
    lon_curr: float, lat_curr: float,
    lon_next: float, lat_next: float
) -> float:
    """
    Calculate SED: perpendicular distance from next point to line (prev -> curr).
    Calculate perpendicular distance of B(next) from baseline A(prev), C(curr).
    Returns distance in meters.
    """
    return cross_track_distance(lon_prev, lat_prev, lon_curr, lat_curr, lon_next, lat_next)


def calculate_sed_local(rows: List[dict]) -> float:
    """
    Calculate local SED for a route segment.
    SED_local = (1/(W-2)) * sum(SED) for k=1 to W-2
    """
    W = len(rows)
    if W < 3:
        return EPSILON_SED

    sed_values = []
    for k in range(1, W - 1):
        p_prev = (float(rows[k - 1]['LON']), float(rows[k - 1]['LAT']))
        p_curr = (float(rows[k]['LON']), float(rows[k]['LAT']))
        p_next = (float(rows[k + 1]['LON']), float(rows[k + 1]['LAT']))
        # Perpendicular distance from midpoint p_curr to line segment (p_prev → p_next)
        sed = calculate_sed(p_prev[0], p_prev[1], p_next[0], p_next[1], p_curr[0], p_curr[1])
        sed_values.append(sed)

    if not sed_values:
        return EPSILON_SED

    return float(np.mean(sed_values))


# =========================
# Velocity/Acceleration Calculation (A2)
# =========================
def calculate_velocities(rows: List[dict], dt_seconds: float = 3600.0) -> List[float]:
    """
    Calculate velocities from positions.
    v_i = ||p_i - p_{i-1}|| / dt
    Returns velocities in m/s (length = W-1).
    """
    velocities = []
    for i in range(1, len(rows)):
        dist = haversine_distance(
            float(rows[i - 1]['LON']), float(rows[i - 1]['LAT']),
            float(rows[i]['LON']), float(rows[i]['LAT'])
        )
        v = dist / dt_seconds
        velocities.append(v)
    return velocities


def calculate_accelerations(velocities: List[float], dt_seconds: float = 3600.0) -> List[float]:
    """
    Calculate accelerations from velocities.
    a_i = (v_i - v_{i-1}) / dt
    Returns accelerations in m/s² (length = len(velocities)-1).
    """
    accelerations = []
    for i in range(1, len(velocities)):
        a = (velocities[i] - velocities[i - 1]) / dt_seconds
        accelerations.append(a)
    return accelerations


def calculate_a_local(rows: List[dict], dt_seconds: float = 3600.0) -> float:
    """
    Calculate local acceleration baseline.
    a_local = (1/(W-1)) * sum(|a_i|)
    """
    velocities = calculate_velocities(rows, dt_seconds)
    if len(velocities) < 2:
        return EPSILON_A

    accelerations = calculate_accelerations(velocities, dt_seconds)
    if not accelerations:
        return EPSILON_A

    return float(np.mean(np.abs(accelerations)))


# =========================
# Angular Velocity Calculation (A2 - heading)
# =========================
def calculate_headings(rows: List[dict]) -> List[float]:
    """Extract headings from COURSE_SIN and COURSE_COS columns."""
    headings = []
    for row in rows:
        h = heading_from_sin_cos(float(row['COURSE_SIN']), float(row['COURSE_COS']))
        headings.append(h)
    return headings


def calculate_omega_values(headings: List[float]) -> List[float]:
    """
    Calculate angular velocities (heading changes).
    omega_i = wrap(h_i - h_{i-1})
    """
    omega_values = []
    for i in range(1, len(headings)):
        omega = wrap_angle(headings[i] - headings[i - 1])
        omega_values.append(omega)
    return omega_values


def calculate_omega_local(rows: List[dict]) -> float:
    """
    Calculate local angular velocity baseline.
    omega_local = (1/(W-1)) * sum(|omega_i|)
    """
    headings = calculate_headings(rows)
    if len(headings) < 2:
        return EPSILON_OMEGA

    omega_values = calculate_omega_values(headings)
    if not omega_values:
        return EPSILON_OMEGA

    return float(np.mean(np.abs(omega_values)))


# =========================
# Cross-track Direction (A1)
# =========================
def get_cross_track_direction(
    lon_prev2: float, lat_prev2: float,
    lon_prev1: float, lat_prev1: float,
    side: int = 1
) -> Tuple[float, float]:
    """
    Calculate cross-track unit direction vector.
    u = (p_{i-1} - p_{i-2}) / ||p_{i-1} - p_{i-2}||
    u_perp = (-u_y, u_x) for left, (u_y, -u_x) for right

    Returns (bearing for perpendicular direction).
    side: 1 for left (+90°), -1 for right (-90°)
    """
    brng = bearing(lon_prev2, lat_prev2, lon_prev1, lat_prev1)
    perp_bearing = (brng + side * 90) % 360
    return perp_bearing


# =========================
# CSV Iteration
# =========================
def iter_routes_csv(csv_path: str) -> Iterator[Tuple[str, List[dict], List[str]]]:
    """
    Iterate over routes in CSV file.
    Yields (route_id, rows, fieldnames) for each route.
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        current_route_id = None
        current_rows = []

        for row in reader:
            route_id = row['ROUTE_ID']

            if current_route_id is None:
                current_route_id = route_id

            if route_id != current_route_id:
                yield current_route_id, current_rows, fieldnames
                current_route_id = route_id
                current_rows = []

            current_rows.append(row)

        if current_rows:
            yield current_route_id, current_rows, fieldnames


# =========================
# Plan Loading
# =========================
def load_plans(injected_dir: str) -> Dict[Tuple[str, str], InjectionPlan]:
    """
    Load A1/A2 injection plans from JSON files.
    Returns dict mapping (route_id, anomaly_type) -> InjectionPlan.

    Filename format: qwen_output_route_{ROUTE_ID}_{A1|A2}_{timestamp}.json
    ROUTE_ID can contain underscores (e.g. "2018CargoApr1-0"), so we parse
    by finding the anomaly type marker (A1/A2) from the right side.
    """
    import re
    plans = {}
    patterns = [
        os.path.join(injected_dir, "qwen_output_route_*_A1_*.json"),
        os.path.join(injected_dir, "qwen_output_route_*_A2_*.json"),
    ]

    rejected = 0
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                if 'K' not in data or 'scores' not in data:
                    continue

                # Extract route_id from filename
                # Format: qwen_output_route_{ROUTE_ID}_{A1|A2}_{timestamp}.json
                basename = os.path.basename(filepath)
                m = re.match(r"qwen_output_route_(.+)_(A[12])_\d{8}-\d{6}-\d{6}\.json$", basename)
                if not m:
                    continue
                route_id = m.group(1)
                anomaly_type = data.get('anomaly_type', m.group(2))

                # Reject unusable score vectors rather than injecting garbage.
                # top_k_indices() on a constant vector returns an arbitrary
                # (last-K) slice, which would look like a successful injection.
                scores = data['scores']
                bad = None
                if not isinstance(scores, list) or not scores:
                    bad = "scores missing or empty"
                elif isinstance(data.get('T'), int) and len(scores) != data['T']:
                    bad = f"scores length {len(scores)} != T {data['T']}"
                elif max(scores) - min(scores) < 1e-9:
                    bad = "scores are degenerate (all values equal)"
                if bad is not None:
                    print(f"Warning: skipping {basename}: {bad}")
                    rejected += 1
                    continue

                plan = InjectionPlan(
                    route_id=route_id,
                    anomaly_type=anomaly_type,
                    T=data['T'],
                    K=data['K'],
                    scores=data['scores']
                )
                plans[(route_id, anomaly_type)] = plan
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")

    if rejected:
        print(f"Warning: {rejected} plan(s) rejected as unusable; re-run `omad score` to regenerate them.")

    return plans


def load_a3_plans(injected_dir: str, fallback_dir: Optional[str] = None) -> Dict[str, A3Plan]:
    """
    Load A3 injection plans from JSON files.
    Returns dict mapping route_id -> A3Plan.
    """
    import re
    plans = {}

    def load_from_dir(directory: str):
        if not os.path.isdir(directory):
            return
        pattern = os.path.join(directory, "qwen_output_route_*_A3*.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Extract route_id: qwen_output_route_{ROUTE_ID}_A3_{timestamp}.json
                basename = os.path.basename(filepath)
                m = re.match(r"qwen_output_route_(.+)_A3_\d{8}-\d{6}-\d{6}\.json$", basename)
                if not m:
                    continue
                route_id = m.group(1)

                plan = A3Plan(
                    route_id=route_id,
                    scores=data['scores']
                )
                plans[route_id] = plan
            except Exception as e:
                print(f"Warning: Failed to load A3 plan {filepath}: {e}")

    load_from_dir(injected_dir)
    if fallback_dir and not plans:
        load_from_dir(fallback_dir)

    return plans
