from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
from manim import Arrow, Line, Text, VGroup

from .registry import register
from .types import BuildResult, World2DContext, World3DContext


AXIS_COLOR = "#ffffff"
GRID_COLOR = "#4a4a4a"
GRID_BOLD_COLOR = "#7a7a7a"
LABEL_COLOR = "#ffffff"


def _range_with_step(raw: Any, fallback: Tuple[float, float], *, default_step: float) -> Tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            start = float(raw[0])
            end = float(raw[1])
            step = float(raw[2]) if len(raw) >= 3 else default_step
            if step == 0:
                step = default_step
            return start, end, abs(step)
        except Exception:
            pass
    return fallback[0], fallback[1], abs(default_step)


def _frange(start: float, end: float, step: float) -> Iterable[float]:
    if step <= 0:
        return []
    count = int((end - start) / step) + 1
    values = []
    for i in range(max(0, count + 2)):
        v = start + i * step
        if v > end + 1e-6:
            break
        values.append(v)
    return values


def _format_tick(value: float) -> str:
    if abs(value) < 1e-8:
        return "0"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _text(label: str, *, font: str, font_size: float, color: str) -> Text:
    return Text(label, font=font, font_size=font_size, color=color)


@register("grid")
@register("graphpaper")
def grid(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    x_step = float(spec.get("x_step", 1.0))
    y_step = float(spec.get("y_step", 1.0))
    bold_every = int(spec.get("bold_every", 5))
    color = str(spec.get("color", GRID_COLOR))
    bold_color = str(spec.get("bold_color", GRID_BOLD_COLOR))
    opacity = float(spec.get("opacity", 0.75))
    bold_opacity = float(spec.get("bold_opacity", 0.9))
    stroke_width = float(spec.get("stroke_width", 1.3))
    bold_width = float(spec.get("bold_width", 1.8))

    group = VGroup()
    x_start = np.floor(ctx.x_min / x_step) * x_step
    y_start = np.floor(ctx.y_min / y_step) * y_step
    x_vals = _frange(x_start, ctx.x_max, x_step)
    y_vals = _frange(y_start, ctx.y_max, y_step)

    for idx, x in enumerate(x_vals):
        start = ctx.to_scene(x, ctx.y_min)
        end = ctx.to_scene(x, ctx.y_max)
        line = Line(start, end)
        is_bold = bold_every > 0 and abs(round(x / x_step)) % bold_every == 0
        line.set_stroke(color=bold_color if is_bold else color, width=bold_width if is_bold else stroke_width)
        line.set_opacity(bold_opacity if is_bold else opacity)
        setattr(line, "_stroke_only", True)
        group.add(line)

    for idx, y in enumerate(y_vals):
        start = ctx.to_scene(ctx.x_min, y)
        end = ctx.to_scene(ctx.x_max, y)
        line = Line(start, end)
        is_bold = bold_every > 0 and abs(round(y / y_step)) % bold_every == 0
        line.set_stroke(color=bold_color if is_bold else color, width=bold_width if is_bold else stroke_width)
        line.set_opacity(bold_opacity if is_bold else opacity)
        setattr(line, "_stroke_only", True)
        group.add(line)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("axes2d")
def axes2d(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    x_min, x_max, x_step = _range_with_step(spec.get("x_range"), (ctx.x_min, ctx.x_max), default_step=1.0)
    y_min, y_max, y_step = _range_with_step(spec.get("y_range"), (ctx.y_min, ctx.y_max), default_step=1.0)
    axis_color = str(spec.get("axis_color", AXIS_COLOR))
    tick_color = str(spec.get("tick_color", AXIS_COLOR))
    label_color = str(spec.get("label_color", LABEL_COLOR))
    tick_size = float(spec.get("tick_size", 0.045))
    font_size = float(spec.get("font_size", 20))
    show_numbers = bool(spec.get("show_numbers", True))
    label_axes = bool(spec.get("label_axes", True))
    show_zero = bool(spec.get("show_zero", True))
    show_zero_once = bool(spec.get("show_zero_once", True))
    show_zero_once = bool(spec.get("show_zero_once", True))

    group = VGroup()
    origin_x = 0.0
    origin_y = 0.0

    x_axis = Line(ctx.to_scene(x_min, origin_y), ctx.to_scene(x_max, origin_y))
    y_axis = Line(ctx.to_scene(origin_x, y_min), ctx.to_scene(origin_x, y_max))
    x_axis.set_stroke(color=axis_color, width=2.6)
    y_axis.set_stroke(color=axis_color, width=2.6)
    setattr(x_axis, "_stroke_only", True)
    setattr(y_axis, "_stroke_only", True)
    group.add(x_axis, y_axis)

    # X ticks and labels
    for x in _frange(x_min, x_max, x_step):
        if not show_zero and abs(x) < 1e-8:
            continue
        pos = ctx.to_scene(x, origin_y)
        tick = Line(pos + np.array([0, -tick_size, 0]), pos + np.array([0, tick_size, 0]))
        tick.set_stroke(color=tick_color, width=2.2)
        setattr(tick, "_stroke_only", True)
        group.add(tick)
        if show_numbers:
            label = _text(_format_tick(x), font=theme.font, font_size=font_size, color=label_color)
            label.next_to(tick, direction=np.array([0, -1, 0]), buff=0.12)
            group.add(label)

    # Y ticks and labels
    for y in _frange(y_min, y_max, y_step):
        if abs(y) < 1e-8 and ((not show_zero) or show_zero_once):
            continue
        pos = ctx.to_scene(origin_x, y)
        tick = Line(pos + np.array([-tick_size, 0, 0]), pos + np.array([tick_size, 0, 0]))
        tick.set_stroke(color=tick_color, width=2.2)
        setattr(tick, "_stroke_only", True)
        group.add(tick)
        if show_numbers:
            label = _text(_format_tick(y), font=theme.font, font_size=font_size, color=label_color)
            label.next_to(tick, direction=np.array([-1, 0, 0]), buff=0.12)
            group.add(label)

    if label_axes:
        x_label = _text("x", font=theme.font, font_size=font_size + 4, color=label_color)
        y_label = _text("y", font=theme.font, font_size=font_size + 4, color=label_color)
        x_label.next_to(x_axis, direction=np.array([1, 0, 0]), buff=0.2)
        y_label.next_to(y_axis, direction=np.array([0, 1, 0]), buff=0.2)
        group.add(x_label, y_label)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("number_line")
def number_line(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    x_min, x_max, x_step = _range_with_step(spec.get("range"), (ctx.x_min, ctx.x_max), default_step=1.0)
    y = float(spec.get("y", 0.0))
    line_color = str(spec.get("line_color", AXIS_COLOR))
    tick_color = str(spec.get("tick_color", AXIS_COLOR))
    label_color = str(spec.get("label_color", LABEL_COLOR))
    tick_size = float(spec.get("tick_size", 0.045))
    font_size = float(spec.get("font_size", 20))
    show_numbers = bool(spec.get("show_numbers", True))
    show_zero = bool(spec.get("show_zero", True))
    label = str(spec.get("label", "")).strip()

    group = VGroup()
    axis = Line(ctx.to_scene(x_min, y), ctx.to_scene(x_max, y))
    axis.set_stroke(color=line_color, width=2)
    setattr(axis, "_stroke_only", True)
    group.add(axis)

    for x in _frange(x_min, x_max, x_step):
        if not show_zero and abs(x) < 1e-8:
            continue
        pos = ctx.to_scene(x, y)
        tick = Line(pos + np.array([0, -tick_size, 0]), pos + np.array([0, tick_size, 0]))
        tick.set_stroke(color=tick_color, width=2)
        setattr(tick, "_stroke_only", True)
        group.add(tick)
        if show_numbers:
            num = _text(_format_tick(x), font=theme.font, font_size=font_size, color=label_color)
            num.next_to(tick, direction=np.array([0, -1, 0]), buff=0.12)
            group.add(num)

    if label:
        label_m = _text(label, font=theme.font, font_size=font_size + 4, color=label_color)
        label_m.next_to(axis, direction=np.array([1, 0, 0]), buff=0.2)
        group.add(label_m)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("reference_frame")
def reference_frame(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    axis_color = str(spec.get("axis_color", AXIS_COLOR))
    label_color = str(spec.get("label_color", LABEL_COLOR))
    axis_length = spec.get("axis_length") or [ctx.x_max - ctx.x_min, ctx.y_max - ctx.y_min]
    try:
        axis_len_x = float(axis_length[0])
        axis_len_y = float(axis_length[1])
    except Exception:
        axis_len_x = ctx.x_max - ctx.x_min
        axis_len_y = ctx.y_max - ctx.y_min

    origin = spec.get("origin") or [0.0, 0.0]
    try:
        ox = float(origin[0])
        oy = float(origin[1])
    except Exception:
        ox, oy = 0.0, 0.0

    group = VGroup()
    x_axis = Arrow(
        ctx.to_scene(ox - axis_len_x / 2, oy),
        ctx.to_scene(ox + axis_len_x / 2, oy),
        buff=0,
        stroke_width=2,
        color=axis_color,
    )
    y_axis = Arrow(
        ctx.to_scene(ox, oy - axis_len_y / 2),
        ctx.to_scene(ox, oy + axis_len_y / 2),
        buff=0,
        stroke_width=2,
        color=axis_color,
    )
    group.add(x_axis, y_axis)

    label_x = str(spec.get("label_x", "x"))
    label_y = str(spec.get("label_y", "y"))
    lx = _text(label_x, font=theme.font, font_size=28, color=label_color)
    ly = _text(label_y, font=theme.font, font_size=28, color=label_color)
    lx.next_to(x_axis, direction=np.array([1, 0, 0]), buff=0.15)
    ly.next_to(y_axis, direction=np.array([0, 1, 0]), buff=0.15)
    group.add(lx, ly)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


def _axis_tick(
    pos: np.ndarray,
    direction: np.ndarray,
    size: float,
    *,
    color: str,
    stroke_width: float,
) -> Line:
    if np.linalg.norm(direction[:2]) < 1e-6:
        perp = np.array([0.0, 1.0, 0.0])
    else:
        dx, dy = direction[0], direction[1]
        perp = np.array([-dy, dx, 0.0])
        perp = perp / max(1e-6, np.linalg.norm(perp))
    start = pos - perp * size
    end = pos + perp * size
    tick = Line(start, end)
    tick.set_stroke(color=color, width=stroke_width)
    return tick


def _arrowhead_lines(
    start: np.ndarray,
    end: np.ndarray,
    *,
    tip_length: float,
    tip_angle_deg: float,
) -> list[Line]:
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


@register("axes3d")
def axes3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    x_min, x_max, x_step = _range_with_step(spec.get("x_range"), (ctx.x_min, ctx.x_max), default_step=1.0)
    y_min, y_max, y_step = _range_with_step(spec.get("y_range"), (ctx.y_min, ctx.y_max), default_step=1.0)
    z_min, z_max, z_step = _range_with_step(spec.get("z_range"), (ctx.z_min, ctx.z_max), default_step=1.0)
    axis_color = str(spec.get("axis_color", AXIS_COLOR))
    tick_color = str(spec.get("tick_color", AXIS_COLOR))
    label_color = str(spec.get("label_color", LABEL_COLOR))
    axis_arrows = bool(spec.get("axis_arrows", True))
    grid_planes = spec.get("grid_planes")
    grid_step = float(spec.get("grid_step", 1.0))
    grid_bold_every = int(spec.get("grid_bold_every", 5))
    grid_color = str(spec.get("grid_color", GRID_COLOR))
    grid_bold_color = str(spec.get("grid_bold_color", GRID_BOLD_COLOR))
    grid_opacity = float(spec.get("grid_opacity", 0.35))
    grid_bold_opacity = float(spec.get("grid_bold_opacity", 0.55))
    tick_size = float(spec.get("tick_size", 0.04))
    font_size = float(spec.get("font_size", 18))
    show_numbers_raw = spec.get("show_numbers", True)
    show_x = True
    show_y = True
    show_z = False
    if isinstance(show_numbers_raw, dict):
        show_x = bool(show_numbers_raw.get("x", True))
        show_y = bool(show_numbers_raw.get("y", True))
        show_z = bool(show_numbers_raw.get("z", False))
    else:
        flag = bool(show_numbers_raw)
        show_x = flag
        show_y = flag
        show_z = flag
    if "show_numbers_x" in spec:
        show_x = bool(spec.get("show_numbers_x"))
    if "show_numbers_y" in spec:
        show_y = bool(spec.get("show_numbers_y"))
    if "show_numbers_z" in spec:
        show_z = bool(spec.get("show_numbers_z"))
    if "show_z_numbers" in spec:
        show_z = bool(spec.get("show_z_numbers"))
    label_axes = bool(spec.get("label_axes", True))
    show_zero = bool(spec.get("show_zero", True))
    show_zero_once = bool(spec.get("show_zero_once", True))
    tip_length = float(spec.get("tip_length", 0.12))
    tip_angle_deg = float(spec.get("tip_angle_deg", 25.0))
    label_offsets = spec.get("label_offsets", {})
    offset_x = float(label_offsets.get("x", 1)) if isinstance(label_offsets, dict) else 1.0
    offset_y = float(label_offsets.get("y", -1)) if isinstance(label_offsets, dict) else -1.0
    offset_z = float(label_offsets.get("z", 1)) if isinstance(label_offsets, dict) else 1.0

    group = VGroup()
    grid_group = VGroup()
    origin = ctx.project(0.0, 0.0, 0.0)
    x_start = ctx.project(x_min, 0.0, 0.0)
    x_end = ctx.project(x_max, 0.0, 0.0)
    y_start = ctx.project(0.0, y_min, 0.0)
    y_end = ctx.project(0.0, y_max, 0.0)
    z_start = ctx.project(0.0, 0.0, z_min)
    z_end = ctx.project(0.0, 0.0, z_max)

    x_axis = Line(x_start, x_end)
    y_axis = Line(y_start, y_end)
    z_axis = Line(z_start, z_end)
    for axis in (x_axis, y_axis, z_axis):
        axis.set_stroke(color=axis_color, width=2.6)
        setattr(axis, "_stroke_only", True)

    if axis_arrows:
        for start, end in ((x_start, x_end), (y_start, y_end), (z_start, z_end)):
            for tip in _arrowhead_lines(start, end, tip_length=tip_length, tip_angle_deg=tip_angle_deg):
                tip.set_stroke(color=axis_color, width=2.6)
                setattr(tip, "_stroke_only", True)
                group.add(tip)
    # Grid planes behind axes
    if grid_planes:
        planes = grid_planes
        if grid_planes is True:
            planes = ["xy"]
        if isinstance(planes, str):
            planes = [planes]
        if isinstance(planes, list):
            for plane in planes:
                _draw_grid_plane(
                    grid_group,
                    ctx,
                    plane=str(plane).lower(),
                    step=grid_step,
                    bold_every=grid_bold_every,
                    color=grid_color,
                    bold_color=grid_bold_color,
                    opacity=grid_opacity,
                    bold_opacity=grid_bold_opacity,
                )
    group.add(grid_group, x_axis, y_axis, z_axis)

    def _axis_perp(basis: np.ndarray) -> np.ndarray:
        dx, dy = basis[0], basis[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return np.array([0.0, 1.0, 0.0])
        perp = np.array([-dy, dx, 0.0])
        return perp / max(1e-6, np.linalg.norm(perp))

    perp_x = _axis_perp(ctx.basis_x)
    perp_y = _axis_perp(ctx.basis_y)
    perp_z = _axis_perp(ctx.basis_z)

    # X ticks
    for x in _frange(x_min, x_max, x_step):
        if not show_zero and abs(x) < 1e-8:
            continue
        pos = ctx.project(x, 0.0, 0.0)
        tick = _axis_tick(pos, ctx.basis_x, tick_size, color=tick_color, stroke_width=2.2)
        setattr(tick, "_stroke_only", True)
        group.add(tick)
        if show_x:
            label = _text(_format_tick(x), font=theme.font, font_size=font_size, color=label_color)
            label.next_to(tick, direction=perp_x * offset_x, buff=0.08)
            group.add(label)

    # Y ticks
    for y in _frange(y_min, y_max, y_step):
        if abs(y) < 1e-8 and ((not show_zero) or show_zero_once):
            continue
        pos = ctx.project(0.0, y, 0.0)
        tick = _axis_tick(pos, ctx.basis_y, tick_size, color=tick_color, stroke_width=2.2)
        setattr(tick, "_stroke_only", True)
        group.add(tick)
        if show_y:
            label = _text(_format_tick(y), font=theme.font, font_size=font_size, color=label_color)
            label.next_to(tick, direction=perp_y * offset_y, buff=0.08)
            group.add(label)

    # Z ticks
    for z in _frange(z_min, z_max, z_step):
        if abs(z) < 1e-8 and ((not show_zero) or show_zero_once):
            continue
        pos = ctx.project(0.0, 0.0, z)
        tick = _axis_tick(pos, ctx.basis_z, tick_size, color=tick_color, stroke_width=2.2)
        setattr(tick, "_stroke_only", True)
        group.add(tick)
        if show_z:
            label = _text(_format_tick(z), font=theme.font, font_size=font_size, color=label_color)
            label.next_to(tick, direction=perp_z * offset_z, buff=0.08)
            group.add(label)

    if label_axes:
        x_label = _text("x", font=theme.font, font_size=font_size + 4, color=label_color)
        y_label = _text("y", font=theme.font, font_size=font_size + 4, color=label_color)
        z_label = _text("z", font=theme.font, font_size=font_size + 4, color=label_color)
        x_label.next_to(x_axis, direction=np.array([1, 0, 0]), buff=0.15)
        y_label.next_to(y_axis, direction=np.array([0, 1, 0]), buff=0.15)
        z_label.next_to(z_axis, direction=np.array([0, 1, 0]), buff=0.15)
        group.add(x_label, y_label, z_label)

    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


@register("world3d")
def world3d(*, spec: Dict[str, Any], ctx: World3DContext, theme, **_) -> BuildResult:
    # world3d is a container in builder; keep a small fallback label here.
    label = _text("World3D", font=theme.font, font_size=24, color=LABEL_COLOR)
    group = VGroup(label)
    _apply_visibility(spec, group)
    return BuildResult(group, id_map=_register_id(spec, group))


def _draw_grid_plane(
    group: VGroup,
    ctx: World3DContext,
    *,
    plane: str,
    step: float,
    bold_every: int,
    color: str,
    bold_color: str,
    opacity: float,
    bold_opacity: float,
) -> None:
    if step <= 0:
        return
    if plane == "xy":
        x_min, x_max = ctx.x_min, ctx.x_max
        y_min, y_max = ctx.y_min, ctx.y_max
        z = 0.0
        for x in _frange(x_min, x_max, step):
            is_bold = bold_every > 0 and abs(round(x / step)) % bold_every == 0
            start = ctx.project(x, y_min, z)
            end = ctx.project(x, y_max, z)
            line = Line(start, end)
            line.set_stroke(color=bold_color if is_bold else color, width=1.3)
            line.set_opacity(bold_opacity if is_bold else opacity)
            setattr(line, "_stroke_only", True)
            group.add(line)
        for y in _frange(y_min, y_max, step):
            is_bold = bold_every > 0 and abs(round(y / step)) % bold_every == 0
            start = ctx.project(x_min, y, z)
            end = ctx.project(x_max, y, z)
            line = Line(start, end)
            line.set_stroke(color=bold_color if is_bold else color, width=1.3)
            line.set_opacity(bold_opacity if is_bold else opacity)
            setattr(line, "_stroke_only", True)
            group.add(line)
    elif plane == "xz":
        x_min, x_max = ctx.x_min, ctx.x_max
        z_min, z_max = ctx.z_min, ctx.z_max
        y = 0.0
        for x in _frange(x_min, x_max, step):
            is_bold = bold_every > 0 and abs(round(x / step)) % bold_every == 0
            start = ctx.project(x, y, z_min)
            end = ctx.project(x, y, z_max)
            line = Line(start, end)
            line.set_stroke(color=bold_color if is_bold else color, width=1.3)
            line.set_opacity(bold_opacity if is_bold else opacity)
            setattr(line, "_stroke_only", True)
            group.add(line)
        for z in _frange(z_min, z_max, step):
            is_bold = bold_every > 0 and abs(round(z / step)) % bold_every == 0
            start = ctx.project(x_min, y, z)
            end = ctx.project(x_max, y, z)
            line = Line(start, end)
            line.set_stroke(color=bold_color if is_bold else color, width=1.3)
            line.set_opacity(bold_opacity if is_bold else opacity)
            setattr(line, "_stroke_only", True)
            group.add(line)
    elif plane == "yz":
        y_min, y_max = ctx.y_min, ctx.y_max
        z_min, z_max = ctx.z_min, ctx.z_max
        x = 0.0
        for y in _frange(y_min, y_max, step):
            is_bold = bold_every > 0 and abs(round(y / step)) % bold_every == 0
            start = ctx.project(x, y, z_min)
            end = ctx.project(x, y, z_max)
            line = Line(start, end)
            line.set_stroke(color=bold_color if is_bold else color, width=1.3)
            line.set_opacity(bold_opacity if is_bold else opacity)
            setattr(line, "_stroke_only", True)
            group.add(line)
        for z in _frange(z_min, z_max, step):
            is_bold = bold_every > 0 and abs(round(z / step)) % bold_every == 0
            start = ctx.project(x, y_min, z)
            end = ctx.project(x, y_max, z)
            line = Line(start, end)
            line.set_stroke(color=bold_color if is_bold else color, width=1.3)
            line.set_opacity(bold_opacity if is_bold else opacity)
            setattr(line, "_stroke_only", True)
            group.add(line)


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
