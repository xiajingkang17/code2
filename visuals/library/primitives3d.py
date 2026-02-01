from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from manim import Arrow, Dot, Line, Polygon, Text, VGroup, VMobject

from .registry import register
from .types import BuildResult, World3DContext


DEFAULT_COLOR = "#ffffff"


def _world3d_point(ctx: World3DContext, value: Any, fallback=(0.0, 0.0, 0.0)) -> np.ndarray:
    try:
        x, y, z = value
    except Exception:
        x, y, z = fallback
    return ctx.project(float(x), float(y), float(z))


def _world3d_points(ctx: World3DContext, values: Iterable[Any]) -> List[np.ndarray]:
    points: List[np.ndarray] = []
    for item in values or []:
        points.append(_world3d_point(ctx, item))
    return points


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


@register("point3d")
def point3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    pos = _world3d_point(ctx, spec.get("pos", (0.0, 0.0, 0.0)))
    radius = float(spec.get("radius", 0.06))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.2))
    label = str(spec.get("label", "")).strip()
    font_size = float(spec.get("font_size", 24))
    dot = Dot(point=pos, radius=radius, color=color)
    dot.set_fill(opacity=0)
    dot.set_stroke(color=color, width=stroke_width)
    _stroke_only(dot)
    group = VGroup(dot)
    if label:
        text = Text(label, font=theme.font, font_size=font_size, color=color)
        text.next_to(dot, direction=np.array([1, 0, 0]), buff=0.08)
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("line3d")
def line3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    start = _world3d_point(ctx, spec.get("start", (-1.0, 0.0, 0.0)))
    end = _world3d_point(ctx, spec.get("end", (1.0, 0.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    line = Line(start, end)
    line.set_stroke(color=color, width=stroke_width)
    _stroke_only(line)
    group = VGroup(line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("polyline3d")
def polyline3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    points = _world3d_points(ctx, spec.get("points") or [(-1, 0, 0), (0, 1, 0), (1, 0, 0)])
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    if len(points) >= 2:
        path = VMobject()
        path.set_points_as_corners(points)
    else:
        path = Line(ctx.project(-1, 0, 0), ctx.project(1, 0, 0))
    path.set_stroke(color=color, width=stroke_width)
    _stroke_only(path)
    group = VGroup(path)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("polygon3d")
def polygon3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    points = _world3d_points(ctx, spec.get("points") or [(-1, 0, 0), (1, 0, 0), (0, 1, 0)])
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    poly = Polygon(*points)
    poly.set_stroke(color=color, width=stroke_width)
    poly.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(poly)
    group = VGroup(poly)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("vector3d")
def vector3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    start = _world3d_point(ctx, spec.get("start", (0.0, 0.0, 0.0)))
    end = _world3d_point(ctx, spec.get("end", (1.0, 0.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    buff = float(spec.get("buff", 0.0))
    arrow = Arrow(start, end, buff=buff)
    arrow.set_stroke(color=color, width=stroke_width)
    arrow.set_fill(opacity=0)
    _stroke_only(arrow)
    group = VGroup(arrow)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("text3d")
def text3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    pos = _world3d_point(ctx, spec.get("pos", (0.0, 0.0, 0.0)))
    content = str(spec.get("text", ""))
    color = str(spec.get("color", DEFAULT_COLOR))
    font_size = float(spec.get("font_size", 24))
    text = Text(content, font=theme.font, font_size=font_size, color=color)
    text.move_to(pos)
    group = VGroup(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("cube")
def cube(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    center = spec.get("center", (0.0, 0.0, 0.0))
    try:
        cx, cy, cz = center
    except Exception:
        cx, cy, cz = 0.0, 0.0, 0.0
    size = float(spec.get("size", 1.5))
    half = size / 2.0
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))

    vertices = [
        (cx - half, cy - half, cz - half),
        (cx + half, cy - half, cz - half),
        (cx + half, cy + half, cz - half),
        (cx - half, cy + half, cz - half),
        (cx - half, cy - half, cz + half),
        (cx + half, cy - half, cz + half),
        (cx + half, cy + half, cz + half),
        (cx - half, cy + half, cz + half),
    ]
    proj = [ctx.project(*v) for v in vertices]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    group = VGroup()
    for a, b in edges:
        seg = Line(proj[a], proj[b])
        seg.set_stroke(color=color, width=stroke_width)
        _stroke_only(seg)
        group.add(seg)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))
