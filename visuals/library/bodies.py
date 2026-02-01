from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
from manim import Circle, Dot, Line, Rectangle, RoundedRectangle, Text, VGroup

from .registry import register
from .types import BuildResult, World2DContext


DEFAULT_COLOR = "#ffffff"


def _world_point(ctx: World2DContext, value: Any, fallback=(0.0, 0.0)) -> np.ndarray:
    try:
        x, y = value
    except Exception:
        x, y = fallback
    return ctx.to_scene(float(x), float(y))


def _scale_len(ctx: World2DContext, value: float) -> float:
    return float(value) * min(ctx.scale_x, ctx.scale_y)


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


def _label(text: str, *, font: str, font_size: float, color: str) -> Text:
    return Text(text, font=font, font_size=font_size, color=color)


@register("particle")
def particle(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    radius = _scale_len(ctx, float(spec.get("radius", 0.045)))
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
        text = _label(label, font=theme.font, font_size=font_size, color=color)
        text.next_to(dot, direction=np.array([1, 0, 0]), buff=0.1)
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("block")
def block(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    size = spec.get("size")
    width = float(spec.get("width", 0.8))
    height = float(spec.get("height", width))
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        width, height = float(size[0]), float(size[1])
    elif isinstance(size, (int, float)):
        width = float(size)
        height = float(size)
    color = str(spec.get("color", DEFAULT_COLOR))
    fill_opacity = float(spec.get("fill_opacity", 1.0))
    stroke_width = float(spec.get("stroke_width", 2.6))
    corner = float(spec.get("corner_radius", 0.06))
    font_size = float(spec.get("font_size", 26))
    label = str(spec.get("label", "")).strip()

    rect = RoundedRectangle(
        width=width * ctx.scale_x,
        height=height * ctx.scale_y,
        corner_radius=_scale_len(ctx, corner),
    )
    rect.set_stroke(color=color, width=stroke_width)
    rect.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(rect)
    rect.move_to(pos)
    angle_deg = float(spec.get("angle_deg", spec.get("angle", 0.0)))
    if angle_deg:
        rect.rotate(np.deg2rad(angle_deg))

    group = VGroup(rect)
    if label:
        text = _label(label, font=theme.font, font_size=font_size, color=color)
        text.next_to(rect, direction=np.array([0, -1, 0]), buff=0.12)
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("ball")
def ball(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    radius = _scale_len(ctx, float(spec.get("radius", 0.35)))
    color = str(spec.get("color", DEFAULT_COLOR))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    stroke_width = float(spec.get("stroke_width", 2.6))
    label = str(spec.get("label", "")).strip()
    font_size = float(spec.get("font_size", 26))

    circ = Circle(radius=radius)
    circ.set_stroke(color=color, width=stroke_width)
    circ.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(circ)
    circ.move_to(pos)

    group = VGroup(circ)
    if label:
        text = _label(label, font=theme.font, font_size=font_size, color=color)
        text.next_to(circ, direction=np.array([0, -1, 0]), buff=0.12)
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("cart")
def cart(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    width = float(spec.get("width", 1.6))
    height = float(spec.get("height", 0.45))
    wheel_radius = float(spec.get("wheel_radius", 0.14))
    wheel_out = float(spec.get("wheel_out", 0.08))
    color = str(spec.get("color", DEFAULT_COLOR))
    wheel_color = str(spec.get("wheel_color", color))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    stroke_width = float(spec.get("stroke_width", 2.6))
    wheel_stroke = float(spec.get("wheel_stroke_width", stroke_width + 0.8))

    body = Rectangle(width=width * ctx.scale_x, height=height * ctx.scale_y)
    body.set_stroke(color=color, width=stroke_width)
    body.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(body)
    body_offset = _scale_len(ctx, wheel_radius * 0.9)
    body.move_to(pos + np.array([0.0, body_offset, 0.0]))

    wheel_r = _scale_len(ctx, wheel_radius)
    wheel_left = Circle(radius=wheel_r)
    wheel_right = Circle(radius=wheel_r)
    for wheel in (wheel_left, wheel_right):
        wheel.set_stroke(color=wheel_color, width=wheel_stroke)
        wheel.set_fill(opacity=0)
        _stroke_only(wheel)
    wheel_y = pos[1] - _scale_len(ctx, wheel_out)
    wheel_left.move_to(pos + np.array([-width * ctx.scale_x * 0.25, -_scale_len(ctx, wheel_out), 0.0]))
    wheel_right.move_to(pos + np.array([width * ctx.scale_x * 0.25, -_scale_len(ctx, wheel_out), 0.0]))

    base_line = Line(
        pos + np.array([-width * ctx.scale_x * 0.5, -_scale_len(ctx, wheel_out), 0.0]),
        pos + np.array([width * ctx.scale_x * 0.5, -_scale_len(ctx, wheel_out), 0.0]),
    )
    base_line.set_stroke(color=color, width=stroke_width)
    _stroke_only(base_line)

    group = VGroup(body, base_line, wheel_left, wheel_right)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("rod")
def rod(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 3.6))
    start = spec.get("start")
    end = spec.get("end")
    if start is not None and end is not None:
        p1 = _world_point(ctx, start)
        p2 = _world_point(ctx, end)
    else:
        pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
        length = float(spec.get("length", 2.0)) * ctx.scale_x
        angle = float(spec.get("angle_deg", 0.0))
        dx = np.cos(np.deg2rad(angle)) * length / 2
        dy = np.sin(np.deg2rad(angle)) * length / 2
        p1 = pos + np.array([-dx, -dy, 0.0])
        p2 = pos + np.array([dx, dy, 0.0])
    seg = Line(p1, p2)
    seg.set_stroke(color=color, width=stroke_width)
    _stroke_only(seg)
    group = VGroup(seg)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("ring")
def ring(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    radius = _scale_len(ctx, float(spec.get("radius", 0.6)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    circ = Circle(radius=radius)
    circ.set_stroke(color=color, width=stroke_width)
    circ.set_fill(opacity=0)
    _stroke_only(circ)
    circ.move_to(pos)
    group = VGroup(circ)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("disk")
@register("pulleywheel")
def disk(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    radius = _scale_len(ctx, float(spec.get("radius", 0.6)))
    color = str(spec.get("color", DEFAULT_COLOR))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    stroke_width = float(spec.get("stroke_width", 2.6))
    spokes = int(spec.get("spokes", 3))

    outer = Circle(radius=radius)
    outer.set_stroke(color=color, width=stroke_width)
    outer.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(outer)
    outer.move_to(pos)

    hub = Circle(radius=radius * 0.15)
    hub.set_stroke(color=color, width=stroke_width)
    hub.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(hub)
    hub.move_to(pos)

    group = VGroup(outer, hub)
    for i in range(max(0, spokes)):
        angle = 2 * np.pi * i / max(1, spokes)
        dx = np.cos(angle) * radius * 0.85
        dy = np.sin(angle) * radius * 0.85
        spoke = Line(pos, pos + np.array([dx, dy, 0.0]))
        spoke.set_stroke(color=color, width=stroke_width * 0.8)
        _stroke_only(spoke)
        group.add(spoke)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("beam")
def beam(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    length = float(spec.get("length", 2.8))
    thickness = float(spec.get("thickness", 0.2))
    angle = float(spec.get("angle_deg", 0.0))
    color = str(spec.get("color", DEFAULT_COLOR))
    fill_opacity = float(spec.get("fill_opacity", 0.0))
    stroke_width = float(spec.get("stroke_width", 2.6))

    rect = Rectangle(width=length * ctx.scale_x, height=thickness * ctx.scale_y)
    rect.set_stroke(color=color, width=stroke_width)
    rect.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(rect)
    rect.move_to(pos)
    if angle:
        rect.rotate(np.deg2rad(angle))
    group = VGroup(rect)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("platform")
def platform(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    width = float(spec.get("width", 2.5))
    height = float(spec.get("height", 0.25))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    hatch = bool(spec.get("hatch", True))

    half_w = width * ctx.scale_x / 2
    line = Line(pos + np.array([-half_w, 0.0, 0.0]), pos + np.array([half_w, 0.0, 0.0]))
    line.set_stroke(color=color, width=stroke_width)
    _stroke_only(line)
    group = VGroup(line)

    if hatch:
        count = int(spec.get("hatch_count", 6))
        hatch_len = height * ctx.scale_y
        for i in range(count):
            t = (i + 1) / (count + 1)
            x = (t - 0.5) * width * ctx.scale_x
            start = pos + np.array([x - 0.12 * ctx.scale_x, 0.0, 0.0])
            end = start + np.array([0.24 * ctx.scale_x, -hatch_len, 0.0])
            hatch_line = Line(start, end)
            hatch_line.set_stroke(color=color, width=stroke_width * 0.6, opacity=0.8)
            _stroke_only(hatch_line)
            group.add(hatch_line)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("obstacle")
def obstacle(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    size = spec.get("size")
    width = float(spec.get("width", 1.0))
    height = float(spec.get("height", 0.8))
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        width, height = float(size[0]), float(size[1])
    style = str(spec.get("style", "plain")).strip().lower()
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    fill_opacity = float(spec.get("fill_opacity", 0.0))

    rect = Rectangle(width=width * ctx.scale_x, height=height * ctx.scale_y)
    rect.set_stroke(color=color, width=stroke_width)
    rect.set_fill(color=color, opacity=fill_opacity)
    _stroke_only(rect)
    rect.move_to(pos)
    group = VGroup(rect)

    if style in {"wood", "hatch"}:
        count = int(spec.get("hatch_count", 5))
        for i in range(count):
            t = (i + 1) / (count + 1)
            x0 = (t - 0.5) * width * ctx.scale_x
            line = Line(
                rect.get_center() + np.array([x0 - 0.2 * ctx.scale_x, -height * ctx.scale_y / 2, 0.0]),
                rect.get_center() + np.array([x0 + 0.2 * ctx.scale_x, height * ctx.scale_y / 2, 0.0]),
            )
            line.set_stroke(color=color, width=stroke_width * 0.6, opacity=0.8)
            _stroke_only(line)
            group.add(line)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))
