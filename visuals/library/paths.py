from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from manim import Arc, Circle, VGroup, VMobject

from .registry import register
from .types import BuildResult, World2DContext


DEFAULT_COLOR = "#ffffff"


def _world_point(ctx: World2DContext, value: Any, fallback=(0.0, 0.0)) -> np.ndarray:
    try:
        x, y = value
    except Exception:
        x, y = fallback
    return ctx.to_scene(float(x), float(y))


def _register_id(spec: Dict[str, Any], mobj) -> Dict[str, Any]:
    obj_id = spec.get("id")
    if not obj_id:
        return {}
    text = str(obj_id).strip()
    if not text:
        return {}
    return {text: mobj}


def _apply_visibility(spec: Dict[str, Any], mobj) -> None:
    if "opacity" in spec:
        try:
            opacity = float(spec.get("opacity", 1.0))
        except Exception:
            opacity = 1.0
        mobj.set_opacity(max(0.0, min(1.0, opacity)))
        return
    if spec.get("visible") is False:
        mobj.set_opacity(0.0)


def _stroke_only(mobj) -> None:
    setattr(mobj, "_stroke_only", True)


def _to_scene_points(ctx: World2DContext, points: Iterable[Any]) -> List[np.ndarray]:
    scene_pts: List[np.ndarray] = []
    for pt in points:
        try:
            x, y = pt
        except Exception:
            continue
        scene_pts.append(ctx.to_scene(float(x), float(y)))
    return scene_pts


def _safe_eval(expr: str, *, t: np.ndarray) -> np.ndarray:
    safe = {
        "np": np,
        "t": t,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
    }
    return eval(expr, {"__builtins__": {}}, safe)  # noqa: S307 - controlled environment


@register("path_polyline")
def path_polyline(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    points = spec.get("points")
    if not points:
        start = spec.get("start", (-1.0, 0.0))
        end = spec.get("end", (1.0, 0.0))
        points = [start, end]
    scene_pts = _to_scene_points(ctx, points)
    if len(scene_pts) < 2:
        scene_pts = [ctx.to_scene(-1.0, 0.0), ctx.to_scene(1.0, 0.0)]
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.0))
    path = VMobject()
    path.set_points_as_corners(scene_pts)
    path.set_stroke(color=color, width=stroke_width)
    _stroke_only(path)
    group = VGroup(path)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("path_arc")
def path_arc(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 1.2)) * min(ctx.scale_x, ctx.scale_y)
    start_deg = float(spec.get("start_angle_deg", 0.0))
    angle_deg = float(spec.get("angle_deg", 180.0))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.0))
    arc = Arc(radius=radius, start_angle=np.deg2rad(start_deg), angle=np.deg2rad(angle_deg))
    arc.move_to(center)
    arc.set_stroke(color=color, width=stroke_width)
    _stroke_only(arc)
    group = VGroup(arc)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("path_circle")
def path_circle(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 1.2)) * min(ctx.scale_x, ctx.scale_y)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.0))
    circ = Circle(radius=radius)
    circ.move_to(center)
    circ.set_stroke(color=color, width=stroke_width)
    circ.set_fill(opacity=0)
    _stroke_only(circ)
    group = VGroup(circ)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("path_projectile")
def path_projectile(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    origin = spec.get("origin", (0.0, 0.0))
    try:
        x0, y0 = float(origin[0]), float(origin[1])
    except Exception:
        x0, y0 = 0.0, 0.0
    v0 = float(spec.get("v0", 4.0))
    angle_deg = float(spec.get("angle_deg", 45.0))
    g = float(spec.get("g", 9.8))

    t_start = float(spec.get("t_start", 0.0))
    t_end = float(spec.get("t_end", 2.0))
    t_range = spec.get("t_range")
    t_step = None
    if isinstance(t_range, (list, tuple)) and len(t_range) >= 2:
        try:
            t_start = float(t_range[0])
            t_end = float(t_range[1])
            if len(t_range) >= 3:
                t_step = float(t_range[2])
        except Exception:
            pass
    samples = int(spec.get("samples", 60))
    if t_step and t_step > 0:
        t_vals = np.arange(t_start, t_end + 1e-6, t_step)
    else:
        t_vals = np.linspace(t_start, t_end, max(2, samples))

    theta = np.deg2rad(angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    points = []
    for t in t_vals:
        x = x0 + vx * t
        y = y0 + vy * t - 0.5 * g * t * t
        points.append((x, y))

    scene_pts = _to_scene_points(ctx, points)
    if len(scene_pts) < 2:
        scene_pts = [ctx.to_scene(-1.0, 0.0), ctx.to_scene(1.0, 0.0)]
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.0))
    path = VMobject()
    path.set_points_as_corners(scene_pts)
    path.set_stroke(color=color, width=stroke_width)
    _stroke_only(path)
    group = VGroup(path)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("path_parametric")
def path_parametric(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    points = spec.get("points")
    if points:
        scene_pts = _to_scene_points(ctx, points)
    else:
        x_points = spec.get("x_points")
        y_points = spec.get("y_points")
        scene_pts = []
        if isinstance(x_points, (list, tuple)) and isinstance(y_points, (list, tuple)):
            for x, y in zip(x_points, y_points):
                scene_pts.append(ctx.to_scene(float(x), float(y)))
        else:
            expr_x = str(spec.get("expr_x", "")).strip()
            expr_y = str(spec.get("expr_y", "")).strip()
            t_start = float(spec.get("t_start", 0.0))
            t_end = float(spec.get("t_end", 2.0))
            t_range = spec.get("t_range")
            if isinstance(t_range, (list, tuple)) and len(t_range) >= 2:
                try:
                    t_start = float(t_range[0])
                    t_end = float(t_range[1])
                except Exception:
                    pass
            samples = int(spec.get("samples", 60))
            t_vals = np.linspace(t_start, t_end, max(2, samples))
            if expr_x and expr_y:
                try:
                    xs = _safe_eval(expr_x, t=t_vals)
                    ys = _safe_eval(expr_y, t=t_vals)
                    xs = np.asarray(xs, dtype=float)
                    ys = np.asarray(ys, dtype=float)
                    if xs.size == 1:
                        xs = np.full_like(t_vals, float(xs))
                    if ys.size == 1:
                        ys = np.full_like(t_vals, float(ys))
                    for x, y in zip(xs, ys):
                        scene_pts.append(ctx.to_scene(float(x), float(y)))
                except Exception:
                    scene_pts = []

    if len(scene_pts) < 2:
        scene_pts = [ctx.to_scene(-1.0, 0.0), ctx.to_scene(1.0, 0.0)]
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.0))
    path = VMobject()
    path.set_points_as_corners(scene_pts)
    path.set_stroke(color=color, width=stroke_width)
    _stroke_only(path)
    group = VGroup(path)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))
