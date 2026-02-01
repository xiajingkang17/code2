from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from manim import Mobject


@dataclass(frozen=True)
class BuildResult:
    mobject: Mobject
    id_map: Dict[str, Mobject] = field(default_factory=dict)
    warning: Optional[str] = None


@dataclass(frozen=True)
class World2DContext:
    width: float
    height: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    scale_x: float
    scale_y: float

    def to_scene(self, x: float, y: float) -> np.ndarray:
        cx = (self.x_min + self.x_max) / 2.0
        cy = (self.y_min + self.y_max) / 2.0
        sx = (x - cx) * self.scale_x
        sy = (y - cy) * self.scale_y
        return np.array([sx, sy, 0.0], dtype=float)


@dataclass(frozen=True)
class World3DContext:
    width: float
    height: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    basis_x: np.ndarray
    basis_y: np.ndarray
    basis_z: np.ndarray
    origin_scene: np.ndarray

    def project(self, x: float, y: float, z: float) -> np.ndarray:
        return self.origin_scene + x * self.basis_x + y * self.basis_y + z * self.basis_z


def make_world2d_context(
    *,
    width: float,
    height: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    scale_mode: str = "fit",
) -> World2DContext:
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    scale_x = width / x_span
    scale_y = height / y_span
    if str(scale_mode).strip().lower() == "fit":
        scale = min(scale_x, scale_y)
        scale_x = scale
        scale_y = scale
    return World2DContext(
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        scale_x=scale_x,
        scale_y=scale_y,
    )


def make_world3d_context(
    *,
    width: float,
    height: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    basis: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    margin: float = 0.9,
) -> World3DContext:
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    span_x = max(1e-6, x_max - x_min)
    span_y = max(1e-6, y_max - y_min)
    span_z = max(1e-6, z_max - z_min)

    if basis is None:
        bx2 = np.array([1.0, 0.0, 0.0], dtype=float)
        by2 = np.array([0.6, 0.3, 0.0], dtype=float)
        bz2 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        bx2 = np.array([basis[0][0], basis[0][1], 0.0], dtype=float)
        by2 = np.array([basis[1][0], basis[1][1], 0.0], dtype=float)
        bz2 = np.array([basis[2][0], basis[2][1], 0.0], dtype=float)

    proj_w = abs(bx2[0]) * span_x + abs(by2[0]) * span_y + abs(bz2[0]) * span_z
    proj_h = abs(bx2[1]) * span_x + abs(by2[1]) * span_y + abs(bz2[1]) * span_z
    if proj_w <= 1e-6:
        proj_w = 1.0
    if proj_h <= 1e-6:
        proj_h = 1.0
    scale = min(width / proj_w, height / proj_h) * margin

    basis_x = bx2 * scale
    basis_y = by2 * scale
    basis_z = bz2 * scale

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    cz = (z_min + z_max) / 2.0
    origin_scene = -(cx * basis_x + cy * basis_y + cz * basis_z)

    return World3DContext(
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        basis_x=basis_x,
        basis_y=basis_y,
        basis_z=basis_z,
        origin_scene=origin_scene,
    )
