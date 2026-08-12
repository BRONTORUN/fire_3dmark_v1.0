from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Return a 3x3 rotation matrix using the fixed Rz @ Ry @ Rx convention."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz_m @ ry_m @ rx_m


def cone_geometry(r_h: float, segments: int = 64) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Canonical cone: base center at origin, height 1 along +Y."""
    r = max(float(r_h), 1e-6)
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    base = np.column_stack((r * np.cos(angles), np.zeros(segments), r * np.sin(angles)))
    tip = np.array([[0.0, 1.0, 0.0]])
    vertices = np.vstack((base, tip))
    tip_index = segments
    edges: list[tuple[int, int]] = []
    for i in range(segments):
        edges.append((i, (i + 1) % segments))
    step = max(1, segments // 8)
    edges.extend((i, tip_index) for i in range(0, segments, step))
    return vertices, edges


def cuboid_geometry(l_h: float, w_h: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Canonical cuboid: base centered at origin, height 1 along +Y."""
    lx = max(float(l_h), 1e-6) / 2.0
    wz = max(float(w_h), 1e-6) / 2.0
    vertices = np.array(
        [
            [-lx, 0.0, -wz], [lx, 0.0, -wz], [lx, 0.0, wz], [-lx, 0.0, wz],
            [-lx, 1.0, -wz], [lx, 1.0, -wz], [lx, 1.0, wz], [-lx, 1.0, wz],
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return vertices, edges


def project_weak_perspective(
    vertices: np.ndarray,
    rotation: np.ndarray,
    u_norm: float,
    v_norm: float,
    scale_norm: float,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D vertices into image pixels with a fixed weak-perspective model."""
    pts = np.asarray(vertices, dtype=float) @ rotation.T
    scale_px = float(scale_norm) * float(image_height)
    u0 = float(u_norm) * float(image_width)
    v0 = float(v_norm) * float(image_height)
    uv = np.column_stack((u0 + scale_px * pts[:, 0], v0 - scale_px * pts[:, 1]))
    return uv, pts[:, 2]


def convex_hull(points: Iterable[Iterable[float]]) -> np.ndarray:
    """2D convex hull by Andrew's monotonic-chain algorithm."""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def rotation_6d(rotation: np.ndarray) -> list[float]:
    """Return the first two columns of a rotation matrix as a 6D label."""
    r = np.asarray(rotation, dtype=float)
    return [float(x) for x in np.concatenate((r[:, 0], r[:, 1]))]
