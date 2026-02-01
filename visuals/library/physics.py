from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from manim import Arc, Arrow, Line, Polygon, Text, VGroup

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


def _label(text: str, *, font: str, font_size: float, color: str) -> Text:
    return Text(text, font=font, font_size=font_size, color=color)


def _dashed_segments(
    start: np.ndarray, end: np.ndarray, *, dash: float, gap: float
) -> List[Line]:
    vec = end - start
    length = float(np.linalg.norm(vec[:2]))
    if length <= 1e-6:
        return []
    direction = vec / length
    segments = []
    pos = 0.0
    while pos < length:
        seg_len = min(dash, length - pos)
        seg_start = start + direction * pos
        seg_end = start + direction * (pos + seg_len)
        segments.append(Line(seg_start, seg_end))
        pos += dash + gap
    return segments


def _segment_line(start: np.ndarray, end: np.ndarray, *, color: str, stroke_width: float) -> Line:
    line = Line(start, end)
    line.set_stroke(color=color, width=stroke_width)
    _stroke_only(line)
    return line


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


def _perp_unit(vec: np.ndarray) -> np.ndarray:
    if np.linalg.norm(vec[:2]) < 1e-6:
        return np.array([0.0, 1.0, 0.0])
    dx, dy = vec[0], vec[1]
    perp = np.array([-dy, dx, 0.0])
    return perp / max(1e-6, np.linalg.norm(perp[:2]))


def _hatch_lines(
    start: np.ndarray,
    end: np.ndarray,
    *,
    count: int,
    length: float,
    slant: float,
    color: str,
    stroke_width: float,
    side: str = "negative",
) -> List[Line]:
    vec = end - start
    seg_len = float(np.linalg.norm(vec[:2]))
    if seg_len <= 1e-6:
        return []
    direction = vec / max(1e-6, seg_len)
    perp = _perp_unit(direction)
    side_norm = str(side).strip().lower()
    if side_norm in {"negative", "down", "below", "back"}:
        perp = -perp
    lines: List[Line] = []
    count = max(1, int(count))
    for i in range(count):
        t = (i + 1) / (count + 1)
        pos = start + direction * (seg_len * t)
        tip = pos + perp * length - direction * (length * slant)
        hatch = Line(pos, tip)
        hatch.set_stroke(color=color, width=stroke_width)
        _stroke_only(hatch)
        lines.append(hatch)
    return lines


def _line_from_pos_angle(
    ctx: World2DContext, *, pos: Tuple[float, float], length: float, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    center = _world_point(ctx, pos)
    half = length * 0.5
    dx = np.cos(np.deg2rad(angle_deg)) * half * ctx.scale_x
    dy = np.sin(np.deg2rad(angle_deg)) * half * ctx.scale_y
    start = center + np.array([-dx, -dy, 0.0])
    end = center + np.array([dx, dy, 0.0])
    return start, end


@register("dashed_line")
def dashed_line(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = _world_point(ctx, spec.get("start", (-1.0, 0.0)))
    end = _world_point(ctx, spec.get("end", (1.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    dash = float(spec.get("dash", 0.25)) * min(ctx.scale_x, ctx.scale_y)
    gap = float(spec.get("gap", 0.18)) * min(ctx.scale_x, ctx.scale_y)
    group = VGroup()
    for seg in _dashed_segments(start, end, dash=dash, gap=gap):
        seg.set_stroke(color=color, width=stroke_width)
        _stroke_only(seg)
        group.add(seg)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("ground")
def ground(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = _world_point(ctx, spec.get("start", (-4.0, 0.0)))
    end = _world_point(ctx, spec.get("end", (4.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    line = Line(start, end)
    line.set_stroke(color=color, width=stroke_width)
    _stroke_only(line)
    group = VGroup(line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("incline")
def incline(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    length = float(spec.get("length", 4.0))
    angle = float(spec.get("angle_deg", 30.0))
    start = spec.get("start")
    end = spec.get("end")
    if start is not None and end is not None:
        p1 = _world_point(ctx, start)
        p2 = _world_point(ctx, end)
    else:
        base = _world_point(ctx, spec.get("base", (0.0, 0.0)))
        dx = np.cos(np.deg2rad(angle)) * length * ctx.scale_x
        dy = np.sin(np.deg2rad(angle)) * length * ctx.scale_y
        p1 = base
        p2 = base + np.array([dx, dy, 0.0])
    line = Line(p1, p2)
    line.set_stroke(color=color, width=stroke_width)
    _stroke_only(line)
    group = VGroup(line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("track_arc")
def track_arc(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 1.2)) * min(ctx.scale_x, ctx.scale_y)
    start_deg = float(spec.get("start_angle_deg", 0.0))
    angle_deg = float(spec.get("angle_deg", 180.0))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    arc = Arc(
        radius=radius,
        start_angle=np.deg2rad(start_deg),
        angle=np.deg2rad(angle_deg),
    )
    arc.move_to(center)
    arc.set_stroke(color=color, width=stroke_width)
    _stroke_only(arc)
    group = VGroup(arc)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("step")
def step(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    width = float(spec.get("width", 2.0)) * ctx.scale_x
    height = float(spec.get("height", 1.0)) * ctx.scale_y
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    poly = Polygon(
        pos + np.array([-width / 2, -height / 2, 0.0]),
        pos + np.array([width / 2, -height / 2, 0.0]),
        pos + np.array([width / 2, height / 2, 0.0]),
        pos + np.array([-width / 2, height / 2, 0.0]),
    )
    poly.set_stroke(color=color, width=stroke_width)
    poly.set_fill(opacity=0)
    _stroke_only(poly)
    group = VGroup(poly)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("pulley")
def pulley(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 0.6)) * min(ctx.scale_x, ctx.scale_y)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    axle = bool(spec.get("axle", True))
    circle = Arc(radius=radius, angle=2 * np.pi)
    circle.set_stroke(color=color, width=stroke_width)
    circle.set_fill(opacity=0)
    _stroke_only(circle)
    circle.move_to(center)
    group = VGroup(circle)
    if axle:
        hub = Arc(radius=radius * 0.15, angle=2 * np.pi)
        hub.set_stroke(color=color, width=stroke_width)
        hub.set_fill(opacity=0)
        _stroke_only(hub)
        hub.move_to(center)
        group.add(hub)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("spring")
def spring(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = _world_point(ctx, spec.get("start", (-1.0, 0.0)))
    end = _world_point(ctx, spec.get("end", (1.0, 0.0)))
    turns = int(spec.get("turns", 6))
    amp = float(spec.get("amplitude", 0.2)) * min(ctx.scale_x, ctx.scale_y)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    if turns < 2:
        turns = 2
    vec = end - start
    length = float(np.linalg.norm(vec[:2]))
    direction = vec / max(1e-6, length)
    perp = np.array([-direction[1], direction[0], 0.0])
    points = [start]
    for i in range(1, turns):
        t = i / turns
        offset = amp if i % 2 == 1 else -amp
        points.append(start + direction * (length * t) + perp * offset)
    points.append(end)
    group = VGroup()
    for i in range(len(points) - 1):
        seg = Line(points[i], points[i + 1])
        seg.set_stroke(color=color, width=stroke_width)
        _stroke_only(seg)
        group.add(seg)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("force_arrow")
def force_arrow(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = _world_point(ctx, spec.get("start", (0.0, 0.0)))
    end = _world_point(ctx, spec.get("end", (1.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    label = str(spec.get("label", "")).strip()
    font_size = float(spec.get("font_size", 24))
    tip_length = float(spec.get("tip_length", 0.12)) * min(ctx.scale_x, ctx.scale_y)
    tip_angle_deg = float(spec.get("tip_angle_deg", 25.0))
    shaft = _segment_line(start, end, color=color, stroke_width=stroke_width)
    group = VGroup(shaft)
    for tip in _arrowhead_lines(start, end, tip_length=tip_length, tip_angle_deg=tip_angle_deg):
        tip.set_stroke(color=color, width=stroke_width)
        _stroke_only(tip)
        group.add(tip)
    if label:
        text = _label(label, font=theme.font, font_size=font_size, color=color)
        vec = end - start
        length = float(np.linalg.norm(vec[:2]))
        if length < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = vec / length
        perp = np.array([-direction[1], direction[0], 0.0])
        label_pos = str(spec.get("label_pos", "tip")).strip().lower()
        label_back = spec.get("label_back")
        if label_back is None:
            back_ratio = float(spec.get("label_back_ratio", 0.2))
            back = length * back_ratio
        else:
            back = float(label_back)
        if label_pos == "tail":
            base = start + direction * (0.2 * length)
        elif label_pos == "mid":
            base = start + direction * (0.5 * length)
        else:
            base = end - direction * back
        offset = spec.get("label_offset")
        if isinstance(offset, (list, tuple)) and len(offset) >= 2:
            dx, dy = float(offset[0]), float(offset[1])
            pos = base + np.array([dx, dy, 0.0])
        else:
            if offset is None:
                offset_ratio = float(spec.get("label_offset_ratio", 0.18))
                offset_val = length * offset_ratio
            else:
                offset_val = float(offset)
            side = 1.0
            side_raw = spec.get("label_side")
            if isinstance(side_raw, str):
                if side_raw.strip().lower() in {"negative", "down", "left", "below"}:
                    side = -1.0
            elif isinstance(side_raw, (int, float)) and float(side_raw) < 0:
                side = -1.0
            pos = base + perp * offset_val * side
        text.move_to(pos)
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("dimension_arrow")
def dimension_arrow(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = _world_point(ctx, spec.get("start", (-1.0, 0.0)))
    end = _world_point(ctx, spec.get("end", (1.0, 0.0)))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    label = str(spec.get("label", "")).strip()
    font_size = float(spec.get("font_size", 24))

    arrow1 = Arrow(start, end, buff=0)
    arrow2 = Arrow(end, start, buff=0)
    for arrow in (arrow1, arrow2):
        arrow.set_stroke(color=color, width=stroke_width)
        arrow.set_fill(opacity=0)
        _stroke_only(arrow)
    group = VGroup(arrow1, arrow2)
    if label:
        text = _label(label, font=theme.font, font_size=font_size, color=color)
        mid = (start + end) / 2
        text.move_to(mid + np.array([0, -0.2, 0.0]))
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("angle_marker")
def angle_marker(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    vertex = _world_point(ctx, spec.get("vertex", (0.0, 0.0)))
    p1 = _world_point(ctx, spec.get("p1", (-1.0, 0.0)))
    p2 = _world_point(ctx, spec.get("p2", (0.0, 1.0)))
    radius = float(spec.get("radius", 0.6)) * min(ctx.scale_x, ctx.scale_y)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    label = str(spec.get("label", "")).strip()
    font_size = float(spec.get("font_size", 22))

    v1 = p1 - vertex
    v2 = p2 - vertex
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    # normalize to smallest positive sweep
    delta = a2 - a1
    while delta <= -np.pi:
        delta += 2 * np.pi
    while delta > np.pi:
        delta -= 2 * np.pi
    arc = Arc(radius=radius, start_angle=a1, angle=delta)
    arc.move_to(vertex)
    arc.set_stroke(color=color, width=stroke_width)
    _stroke_only(arc)
    group = VGroup(arc)
    if label:
        mid_angle = a1 + delta / 2
        label_pos = vertex + np.array([np.cos(mid_angle), np.sin(mid_angle), 0.0]) * (radius * 1.15)
        text = _label(label, font=theme.font, font_size=font_size, color=color)
        text.move_to(label_pos)
        group.add(text)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("right_angle")
def right_angle(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    vertex = _world_point(ctx, spec.get("vertex", (0.0, 0.0)))
    p1 = _world_point(ctx, spec.get("p1", (1.0, 0.0)))
    p2 = _world_point(ctx, spec.get("p2", (0.0, 1.0)))
    size = float(spec.get("size", 0.4)) * min(ctx.scale_x, ctx.scale_y)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))

    v1 = p1 - vertex
    v2 = p2 - vertex
    v1 = v1 / max(1e-6, np.linalg.norm(v1[:2]))
    v2 = v2 / max(1e-6, np.linalg.norm(v2[:2]))
    corner = vertex + v1 * size + v2 * size
    poly = Polygon(vertex + v1 * size, corner, vertex + v2 * size)
    poly.set_stroke(color=color, width=stroke_width)
    poly.set_fill(opacity=0)
    _stroke_only(poly)
    group = VGroup(poly)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("horizontal_surface")
@register("smooth_surface")
def horizontal_surface(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = spec.get("start")
    end = spec.get("end")
    if start is not None and end is not None:
        p1 = _world_point(ctx, start)
        p2 = _world_point(ctx, end)
    else:
        pos = spec.get("pos", (0.0, 0.0))
        length = float(spec.get("length", 8.0))
        angle = float(spec.get("angle_deg", 0.0))
        p1, p2 = _line_from_pos_angle(ctx, pos=pos, length=length, angle_deg=angle)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    line = _segment_line(p1, p2, color=color, stroke_width=stroke_width)
    group = VGroup(line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("rough_surface")
def rough_surface(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = spec.get("start")
    end = spec.get("end")
    if start is not None and end is not None:
        p1 = _world_point(ctx, start)
        p2 = _world_point(ctx, end)
    else:
        pos = spec.get("pos", (0.0, 0.0))
        length = float(spec.get("length", 8.0))
        angle = float(spec.get("angle_deg", 0.0))
        p1, p2 = _line_from_pos_angle(ctx, pos=pos, length=length, angle_deg=angle)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    tick_count = int(spec.get("tick_count", 10))
    tick_len = float(spec.get("tick_length", 0.2)) * min(ctx.scale_x, ctx.scale_y)
    tick_slant = float(spec.get("tick_slant", 0.3))
    tick_side = str(spec.get("tick_side", "negative"))
    base = _segment_line(p1, p2, color=color, stroke_width=stroke_width)
    group = VGroup(base)

    for tick in _hatch_lines(
        p1,
        p2,
        count=tick_count,
        length=tick_len,
        slant=tick_slant,
        color=color,
        stroke_width=stroke_width * 0.8,
        side=tick_side,
    ):
        group.add(tick)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("inclined_plane")
def inclined_plane(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    hatch = bool(spec.get("hatch", True))
    hatch_count = int(spec.get("hatch_count", 6))
    hatch_len = float(spec.get("hatch_length", 0.18)) * min(ctx.scale_x, ctx.scale_y)
    hatch_slant = float(spec.get("hatch_slant", 0.3))
    hatch_side = str(spec.get("hatch_side", "negative"))

    length = float(spec.get("length", 4.0))
    angle = float(spec.get("angle_deg", 30.0))
    start = spec.get("start")
    end = spec.get("end")
    if start is not None and end is not None:
        p1 = _world_point(ctx, start)
        p2 = _world_point(ctx, end)
    else:
        base = _world_point(ctx, spec.get("base", (0.0, 0.0)))
        dx = np.cos(np.deg2rad(angle)) * length * ctx.scale_x
        dy = np.sin(np.deg2rad(angle)) * length * ctx.scale_y
        p1 = base
        p2 = base + np.array([dx, dy, 0.0])
    line = _segment_line(p1, p2, color=color, stroke_width=stroke_width)
    group = VGroup(line)
    if hatch:
        for hatch_line in _hatch_lines(
            p1,
            p2,
            count=hatch_count,
            length=hatch_len,
            slant=hatch_slant,
            color=color,
            stroke_width=stroke_width * 0.7,
            side=hatch_side,
        ):
            group.add(hatch_line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("wall")
def wall(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    height = float(spec.get("height", 2.5))
    direction = str(spec.get("direction", "up")).strip().lower()
    dy = height * ctx.scale_y
    if direction == "down":
        p1 = pos
        p2 = pos + np.array([0.0, -dy, 0.0])
    else:
        p1 = pos
        p2 = pos + np.array([0.0, dy, 0.0])
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    hatch = bool(spec.get("hatch", True))
    hatch_count = int(spec.get("hatch_count", 6))
    hatch_len = float(spec.get("hatch_length", 0.18)) * min(ctx.scale_x, ctx.scale_y)
    hatch_slant = float(spec.get("hatch_slant", 0.3))
    hatch_side = str(spec.get("hatch_side", "negative"))
    line = _segment_line(p1, p2, color=color, stroke_width=stroke_width)
    group = VGroup(line)
    if hatch:
        for hatch_line in _hatch_lines(
            p1,
            p2,
            count=hatch_count,
            length=hatch_len,
            slant=hatch_slant,
            color=color,
            stroke_width=stroke_width * 0.7,
            side=hatch_side,
        ):
            group.add(hatch_line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("track_line")
@register("trackline")
def track_line(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = spec.get("start")
    end = spec.get("end")
    if start is not None and end is not None:
        p1 = _world_point(ctx, start)
        p2 = _world_point(ctx, end)
    else:
        pos = spec.get("pos", (0.0, 0.0))
        length = float(spec.get("length", 6.0))
        angle = float(spec.get("angle_deg", 0.0))
        p1, p2 = _line_from_pos_angle(ctx, pos=pos, length=length, angle_deg=angle)
    gap = float(spec.get("gap", 0.2)) * min(ctx.scale_x, ctx.scale_y)
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    double = bool(spec.get("double", False))
    vec = p2 - p1
    direction = vec / max(1e-6, np.linalg.norm(vec[:2]))
    perp = _perp_unit(direction)
    if double:
        line1 = _segment_line(p1 + perp * gap, p2 + perp * gap, color=color, stroke_width=stroke_width)
        line2 = _segment_line(p1 - perp * gap, p2 - perp * gap, color=color, stroke_width=stroke_width)
        group = VGroup(line1, line2)
    else:
        line = _segment_line(p1, p2, color=color, stroke_width=stroke_width)
        group = VGroup(line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("loop_the_loop_track")
def loop_the_loop_track(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    center = _world_point(ctx, spec.get("center", (0.0, 0.0)))
    radius = float(spec.get("radius", 1.6)) * min(ctx.scale_x, ctx.scale_y)
    entry = float(spec.get("entry_length", 2.0)) * ctx.scale_x
    exit_len = float(spec.get("exit_length", entry)) * ctx.scale_x
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))

    loop = Arc(radius=radius, angle=2 * np.pi)
    loop.set_stroke(color=color, width=stroke_width)
    loop.set_fill(opacity=0)
    _stroke_only(loop)
    loop.move_to(center)

    bottom = center + np.array([0.0, -radius, 0.0])
    entry_start = bottom + np.array([-entry, 0.0, 0.0])
    entry_line = _segment_line(entry_start, bottom, color=color, stroke_width=stroke_width)
    exit_end = bottom + np.array([exit_len, 0.0, 0.0])
    exit_line = _segment_line(bottom, exit_end, color=color, stroke_width=stroke_width)
    group = VGroup(loop, entry_line, exit_line)
    hatch = bool(spec.get("line_hatch", True))
    hatch_count = int(spec.get("line_hatch_count", 5))
    hatch_len = float(spec.get("line_hatch_length", 0.16)) * min(ctx.scale_x, ctx.scale_y)
    hatch_slant = float(spec.get("line_hatch_slant", 0.3))
    hatch_side = str(spec.get("line_hatch_side", "negative"))
    if hatch:
        for hatch_line in _hatch_lines(
            entry_start,
            bottom,
            count=hatch_count,
            length=hatch_len,
            slant=hatch_slant,
            color=color,
            stroke_width=stroke_width * 0.7,
            side=hatch_side,
        ):
            group.add(hatch_line)
        for hatch_line in _hatch_lines(
            bottom,
            exit_end,
            count=hatch_count,
            length=hatch_len,
            slant=hatch_slant,
            color=color,
            stroke_width=stroke_width * 0.7,
            side=hatch_side,
        ):
            group.add(hatch_line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("table_with_edge")
def table_with_edge(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    start = spec.get("start")
    end = spec.get("end")
    if start is not None and end is not None:
        p1 = _world_point(ctx, start)
        p2 = _world_point(ctx, end)
    else:
        pos = spec.get("pos", (0.0, 0.0))
        length = float(spec.get("length", 6.0))
        p1, p2 = _line_from_pos_angle(ctx, pos=pos, length=length, angle_deg=0.0)
    edge_height = float(spec.get("edge_height", 1.0)) * ctx.scale_y
    edge_side = str(spec.get("edge_side", "right")).strip().lower()
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))
    top = _segment_line(p1, p2, color=color, stroke_width=stroke_width)
    if edge_side == "left":
        edge_start = p1
    else:
        edge_start = p2
    edge_end = edge_start + np.array([0.0, -edge_height, 0.0])
    edge = _segment_line(edge_start, edge_end, color=color, stroke_width=stroke_width)
    group = VGroup(top, edge)
    top_hatch = bool(spec.get("top_hatch", True))
    top_hatch_count = int(spec.get("top_hatch_count", 6))
    top_hatch_len = float(spec.get("top_hatch_length", 0.12)) * min(ctx.scale_x, ctx.scale_y)
    top_hatch_slant = float(spec.get("top_hatch_slant", 0.3))
    top_hatch_side = str(spec.get("top_hatch_side", "negative"))
    if top_hatch:
        for hatch_line in _hatch_lines(
            p1,
            p2,
            count=top_hatch_count,
            length=top_hatch_len,
            slant=top_hatch_slant,
            color=color,
            stroke_width=stroke_width * 0.6,
            side=top_hatch_side,
        ):
            group.add(hatch_line)
    edge_hatch = bool(spec.get("edge_hatch", True))
    hatch_count = int(spec.get("edge_hatch_count", 4))
    hatch_len = float(spec.get("edge_hatch_length", 0.18)) * min(ctx.scale_x, ctx.scale_y)
    hatch_slant = float(spec.get("edge_hatch_slant", 0.3))
    hatch_side = str(spec.get("edge_hatch_side", "negative"))
    if edge_hatch:
        for hatch_line in _hatch_lines(
            edge_start,
            edge_end,
            count=hatch_count,
            length=hatch_len,
            slant=hatch_slant,
            color=color,
            stroke_width=stroke_width * 0.7,
            side=hatch_side,
        ):
            group.add(hatch_line)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("conveyor_belt")
def conveyor_belt(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = spec.get("pos", (0.0, 0.0))
    length = float(spec.get("length", 6.0))
    thickness = float(spec.get("thickness", 0.3))
    direction = float(spec.get("direction", 1.0))
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.4))
    arrow_count = int(spec.get("arrow_count", 4))

    top_start, top_end = _line_from_pos_angle(ctx, pos=pos, length=length, angle_deg=0.0)
    offset = np.array([0.0, -thickness * ctx.scale_y, 0.0])
    bottom_start = top_start + offset
    bottom_end = top_end + offset

    top = _segment_line(top_start, top_end, color=color, stroke_width=stroke_width)
    bottom = _segment_line(bottom_start, bottom_end, color=color, stroke_width=stroke_width)

    group = VGroup(top, bottom)
    if arrow_count > 0:
        vec = top_end - top_start
        length_px = np.linalg.norm(vec[:2])
        direction_vec = vec / max(1e-6, length_px)
        for i in range(arrow_count):
            t = (i + 1) / (arrow_count + 1)
            base = top_start + direction_vec * length_px * t
            head = base + direction_vec * (0.3 * ctx.scale_x * direction)
            arrow = Arrow(base, head, buff=0)
            arrow.set_stroke(color=color, width=stroke_width)
            arrow.set_fill(opacity=0)
            _stroke_only(arrow)
            group.add(arrow)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("wedge")
def wedge(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    width = float(spec.get("width", 2.0)) * ctx.scale_x
    height = float(spec.get("height", 1.0)) * ctx.scale_y
    orientation = str(spec.get("orientation", "right")).strip().lower()
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))

    left = pos + np.array([-width / 2, -height / 2, 0.0])
    right = pos + np.array([width / 2, -height / 2, 0.0])
    if orientation == "left":
        top = pos + np.array([-width / 2, height / 2, 0.0])
    else:
        top = pos + np.array([width / 2, height / 2, 0.0])
    poly = Polygon(left, right, top)
    poly.set_stroke(color=color, width=stroke_width)
    poly.set_fill(opacity=0)
    _stroke_only(poly)
    group = VGroup(poly)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("pulley_support")
def pulley_support(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    pos = _world_point(ctx, spec.get("pos", (0.0, 0.0)))
    height = float(spec.get("height", 2.0)) * ctx.scale_y
    width = float(spec.get("width", 1.2)) * ctx.scale_x
    color = str(spec.get("color", DEFAULT_COLOR))
    stroke_width = float(spec.get("stroke_width", 2.6))

    base_left = pos + np.array([-width / 2, 0.0, 0.0])
    base_right = pos + np.array([width / 2, 0.0, 0.0])
    top = pos + np.array([0.0, height, 0.0])
    base = _segment_line(base_left, base_right, color=color, stroke_width=stroke_width)
    pole = _segment_line(pos, top, color=color, stroke_width=stroke_width)
    head = _segment_line(top + np.array([-width / 3, 0.0, 0.0]), top + np.array([width / 3, 0.0, 0.0]), color=color, stroke_width=stroke_width)

    group = VGroup(base, pole, head)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))
