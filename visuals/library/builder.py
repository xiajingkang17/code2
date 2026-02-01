from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from manim import Mobject, Rectangle, VGroup

from layout import Theme, make_text_mobject
from .registry import get as get_component
from .types import BuildResult, World2DContext, World3DContext, make_world2d_context, make_world3d_context


DEFAULT_X_RANGE: Tuple[float, float] = (-5.0, 5.0)
DEFAULT_Y_RANGE: Tuple[float, float] = (-3.0, 3.0)


def build_from_spec(
    spec: Optional[Dict[str, Any]],
    width: float,
    height: float,
    *,
    theme: Theme,
) -> BuildResult:
    return _build_spec(spec, width=width, height=height, theme=theme, ctx=None)


def _build_spec(
    spec: Optional[Dict[str, Any]],
    *,
    width: float,
    height: float,
    theme: Theme,
    ctx: Optional[Union[World2DContext, World3DContext]],
) -> BuildResult:
    if not spec:
        return BuildResult(_placeholder(width, height, theme=theme, label="Visual"), warning="empty_visual")

    raw = dict(spec)
    ctype = str(raw.pop("type", "")).strip().lower()
    if not ctype:
        return BuildResult(_placeholder(width, height, theme=theme, label="Visual"), warning="missing_type")

    if ctype == "world2d":
        result = _build_world2d(raw, width=width, height=height, theme=theme)
        _attach_angle_base(result.mobject, raw)
        return result
    if ctype == "world3d":
        result = _build_world3d(raw, width=width, height=height, theme=theme)
        _attach_angle_base(result.mobject, raw)
        return result
    if ctype == "group":
        if ctx is None:
            ctx = _infer_world_context(raw, width=width, height=height)
        result = _build_group(raw, width=width, height=height, theme=theme, ctx=ctx)
        _attach_angle_base(result.mobject, raw)
        return result

    if ctx is None:
        if ctype in {"axes3d"}:
            ctx = _infer_world3d_context(raw, width=width, height=height)
        else:
            ctx = _infer_world_context(raw, width=width, height=height)

    builder = get_component(ctype)
    if builder is None:
        return BuildResult(
            _placeholder(width, height, theme=theme, label=f"Unknown: {ctype}"),
            warning="unknown_component",
        )

    result = builder(spec=raw, ctx=ctx, theme=theme)
    _attach_angle_base(result.mobject, raw)
    return result


def _build_world2d(
    spec: Dict[str, Any],
    *,
    width: float,
    height: float,
    theme: Theme,
) -> BuildResult:
    ctx = _infer_world_context(spec, width=width, height=height)
    group = VGroup()
    id_map: Dict[str, Mobject] = {}
    for child in spec.get("children", []) or []:
        result = _build_spec(child, width=width, height=height, theme=theme, ctx=ctx)
        group.add(result.mobject)
        id_map.update(result.id_map)
    _register_id(id_map, spec.get("id"), group)
    return BuildResult(group, id_map=id_map)


def _build_world3d(
    spec: Dict[str, Any],
    *,
    width: float,
    height: float,
    theme: Theme,
) -> BuildResult:
    ctx3d = _infer_world3d_context(spec, width=width, height=height)
    group = VGroup()
    id_map: Dict[str, Mobject] = {}
    for child in spec.get("children", []) or []:
        result = _build_spec(child, width=width, height=height, theme=theme, ctx=ctx3d)
        group.add(result.mobject)
        id_map.update(result.id_map)
    _register_id(id_map, spec.get("id"), group)
    return BuildResult(group, id_map=id_map)


def _build_group(
    spec: Dict[str, Any],
    *,
    width: float,
    height: float,
    theme: Theme,
    ctx: Union[World2DContext, World3DContext],
) -> BuildResult:
    group = VGroup()
    id_map: Dict[str, Mobject] = {}
    for child in spec.get("children", []) or []:
        result = _build_spec(child, width=width, height=height, theme=theme, ctx=ctx)
        group.add(result.mobject)
        id_map.update(result.id_map)
    _register_id(id_map, spec.get("id"), group)
    return BuildResult(group, id_map=id_map)


def _infer_world_context(spec: Dict[str, Any], *, width: float, height: float) -> World2DContext:
    x_range = _parse_range(spec.get("x_range"), fallback=DEFAULT_X_RANGE)
    y_range = _parse_range(spec.get("y_range"), fallback=DEFAULT_Y_RANGE)
    scale_mode = str(spec.get("scale_mode", "fit")).strip().lower()
    return make_world2d_context(
        width=width,
        height=height,
        x_range=x_range,
        y_range=y_range,
        scale_mode=scale_mode,
    )


def _infer_world3d_context(spec: Dict[str, Any], *, width: float, height: float) -> World3DContext:
    x_range = _parse_range(spec.get("x_range"), fallback=DEFAULT_X_RANGE)
    y_range = _parse_range(spec.get("y_range"), fallback=DEFAULT_Y_RANGE)
    z_range = _parse_range(spec.get("z_range"), fallback=(-2.0, 2.0))
    margin = float(spec.get("margin", 0.9))
    basis = spec.get("basis")
    basis_tuple = None
    if isinstance(basis, (list, tuple)) and len(basis) >= 3:
        try:
            basis_tuple = (
                (float(basis[0][0]), float(basis[0][1])),
                (float(basis[1][0]), float(basis[1][1])),
                (float(basis[2][0]), float(basis[2][1])),
            )
        except Exception:
            basis_tuple = None
    return make_world3d_context(
        width=width,
        height=height,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        basis=basis_tuple,
        margin=margin,
    )


def _parse_range(raw: Any, *, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return float(raw[0]), float(raw[1])
        except Exception:
            return fallback
    return fallback


def _register_id(id_map: Dict[str, Mobject], obj_id: Any, mobj: Mobject) -> None:
    if not obj_id or mobj is None:
        return
    text = str(obj_id).strip()
    if not text:
        return
    id_map.setdefault(text, mobj)


def _attach_angle_base(mobj: Mobject, spec: Dict[str, Any]) -> None:
    if mobj is None:
        return
    base_deg = spec.get("angle_base_deg")
    base_rad = spec.get("angle_base")
    value = None
    if base_rad is not None:
        try:
            value = float(base_rad)
        except Exception:
            value = None
    elif base_deg is not None:
        try:
            value = np.deg2rad(float(base_deg))
        except Exception:
            value = None
    if value is None:
        return
    setattr(mobj, "_angle_base", value)


def _placeholder(width: float, height: float, *, theme: Theme, label: str) -> Mobject:
    frame = Rectangle(width=width, height=height)
    frame.set_stroke(width=2, color=theme.accent_color)
    frame.set_fill(opacity=0)
    text = make_text_mobject(label, font=theme.font, font_size=28)
    text.move_to(frame.get_center())
    return VGroup(frame, text)
