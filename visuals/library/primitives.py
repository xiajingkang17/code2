from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from manim import Arc, Circle, Dot, Line, Polygon, Rectangle, Text, VGroup, VMobject

from .registry import register
from .types import BuildResult, World2DContext


DEFAULT_COLOR = "#ffffff"


def _to_point(value: Iterable[float]) -> np.ndarray:
    arr = np.array(list(value), dtype=float)
    if arr.shape == (2,):
        arr = np.array([arr[0], arr[1], 0.0], dtype=float)
    return arr


def _world_point(ctx: World2DContext, value: Any, fallback=(0.0, 0.0)) -> np.ndarray:
    try:
        x, y = value
    except Exception:
        x, y = fallback
    return ctx.to_scene(float(x), float(y))


def _world_points(ctx: World2DContext, values: Iterable[Any]) -> List[np.ndarray]:
    points: List[np.ndarray] = []
    for item in values or []:
        points.append(_world_point(ctx, item))
    return points


def _label_text(label: str, *, font: str, font_size: float, color: str) -> Text:
    return Text(label, font=font, font_size=font_size, color=color)


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


def _arrowhead_lines(
    start: np.ndarray,
    end: np.ndarray,
    *,
    tip_length: float,
    tip_angle_deg: float,
) -> List[Line]:
    vec = end - start
    norm = float(np.linalg.norm(vec[:2]))
    if norm < 1e-6:
        return []
    direction = vec / norm
    back = np.array([-direction[0], -direction[1], 0.0])
    angle = np.deg2rad(tip_angle_deg)
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    left = np.array([back[0] * cos_a - back[1] * sin_a, back[0] * sin_a + back[1] * cos_a, 0.0])
    right = np.array([back[0] * cos_a + back[1] * sin_a, -back[0] * sin_a + back[1] * cos_a, 0.0])
    p1 = end + left * tip_length
    p2 = end + right * tip_length
    return [Line(end, p1), Line(end, p2)]

@register("point")
def point(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    radius = float(spec.get("radius", 0.045))
    color = str(spec.get("color", DEFAULT_COLOR))
    fill_opacity = float(spec.get("fill_opacity", 1.0))
    default_stroke = 0.0 if fill_opacity > 0 else 2.2
    stroke_width = float(spec.get("stroke_width", default_stroke))
    label = str(spec.get("label", "")).strip()
    font_size = float(spec.get("font_size", 26))

    dot = Dot(point=pos, radius=radius, color=color)
    dot.set_fill(color=color, opacity=fill_opacity)
    dot.set_stroke(color=color, width=stroke_width)
    if fill_opacity <= 0:
        _stroke_only(dot)
    group = VGroup(dot)
    if label:
        text = _label_text(label, font=theme.font, font_size=font_size, color=color)
        text.next_to(dot, direction=np.array([1, 0, 0]), buff=0.1)
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("line")
def line(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = _world_point(ctx, spec.get("start", (-1.0, 0.0)))
    end = _world_point(ctx, spec.get("end", (1.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    line_m = Line(start, end)
    line_m.set_stroke(color=color, width=stroke_width)
    _stroke_only(line_m)
    group = VGroup(line_m)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("polyline")
def polyline(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    points = _world_points(ctx, spec.get("points") or [(-1, 0), (0, 1), (1, 0)])
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    if len(points) < 2:
        points = [ctx.to_scene(-1, 0), ctx.to_scene(1, 0)]
    if len(points) >= 2:
        path = VMobject()
        path.set_points_as_corners(points)
    else:
        path = Line(ctx.to_scene(-1, 0), ctx.to_scene(1, 0))
    path.set_stroke(color=color, width=stroke_width)
    _stroke_only(path)
    group = VGroup(path)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("circle")
def circle(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 1.0))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    circ = Circle(radius=radius * ctx.scale_x)
    circ.move_to(center)
    circ.set_stroke(color=color, width=stroke_width)
    circ.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(circ)
    group = VGroup(circ)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("arc")
def arc(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 1.0))
    start_angle = float(spec.get("start_angle_deg", 0.0))
    angle = float(spec.get("angle_deg", 90.0))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    arc_m = Arc(
        radius=radius * ctx.scale_x,
        start_angle=np.deg2rad(start_angle),
        angle=np.deg2rad(angle),
    )
    arc_m.move_to(center)
    arc_m.set_stroke(color=color, width=stroke_width)
    _stroke_only(arc_m)
    group = VGroup(arc_m)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("rectangle")
def rectangle(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    width = float(spec.get("width", 2.0)) * ctx.scale_x
    height = float(spec.get("height", 1.0)) * ctx.scale_y
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    rect = Rectangle(width=width, height=height)
    rect.move_to(center)
    rect.set_stroke(color=color, width=stroke_width)
    rect.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(rect)
    group = VGroup(rect)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("polygon")
def polygon(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    points = _world_points(ctx, spec.get("points") or [(-1, 0), (1, 0), (0, 1)])
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    poly = Polygon(*points)
    poly.set_stroke(color=color, width=stroke_width)
    poly.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(poly)
    group = VGroup(poly)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("text")
def text(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    content = str(spec.get("text", ""))
    color = str(spec.get("color", DEFAULT_COLOR))
    font_size = float(spec.get("font_size", 28))
    text_m = Text(content, font=theme.font, font_size=font_size, color=color)
    text_m.move_to(pos)
    group = VGroup(text_m)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("arrow")
def arrow(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = _world_point(ctx, spec.get("start", (-1.0, 0.0)))
    end = _world_point(ctx, spec.get("end", (1.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    buff = float(spec.get("buff", 0.0)) * min(ctx.scale_x, ctx.scale_y)
    tip_length = float(spec.get("tip_length", 0.12)) * min(ctx.scale_x, ctx.scale_y)
    tip_angle_deg = float(spec.get("tip_angle_deg", 25.0))

    vec = end - start
    length = float(np.linalg.norm(vec[:2]))
    if length > 1e-6 and buff > 0 and length > 2 * buff:
        direction = vec / length
        start = start + direction * buff
        end = end - direction * buff

    shaft = Line(start, end)
    shaft.set_stroke(color=color, width=stroke_width)
    _stroke_only(shaft)
    group = VGroup(shaft)
    for tip in _arrowhead_lines(start, end, tip_length=tip_length, tip_angle_deg=tip_angle_deg):
        tip.set_stroke(color=color, width=stroke_width)
        _stroke_only(tip)
        group.add(tip)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))
