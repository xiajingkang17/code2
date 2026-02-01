from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from manim import VGroup, VMobject

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


def _normalize_normal(vec: np.ndarray, mode: str) -> np.ndarray:
    nx, ny = float(vec[0]), float(vec[1])
    mode = (mode or "up").strip().lower()
    if mode == "up" and ny < 0:
        nx, ny = -nx, -ny
    elif mode == "down" and ny > 0:
        nx, ny = -nx, -ny
    elif mode == "left" and nx > 0:
        nx, ny = -nx, -ny
    elif mode == "right" and nx < 0:
        nx, ny = -nx, -ny
    return np.array([nx, ny, 0.0], dtype=float)


def _build_track_points(
    spec: Dict[str, Any],
    ctx: World2DContext,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    points = spec.get("points")
    if isinstance(points, (list, tuple)) and points:
        scene_pts: List[np.ndarray] = []
        for pt in points:
            scene_pts.append(_world_point(ctx, pt))
        segments: List[Dict[str, Any]] = []
        for idx in range(len(scene_pts) - 1):
            p0 = scene_pts[idx]
            p1 = scene_pts[idx + 1]
            if np.linalg.norm(p1[:2] - p0[:2]) > 1e-6:
                segments.append({"type": "line", "p0": p0, "p1": p1})
        return scene_pts, segments

    segments = spec.get("segments") or []
    scene_pts: List[np.ndarray] = []
    track_segments: List[Dict[str, Any]] = []
    scale_is_uniform = abs(ctx.scale_x - ctx.scale_y) < 1e-6
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        stype = str(seg.get("type", "line")).strip().lower()
        if stype == "line":
            a = seg.get("a", seg.get("start"))
            b = seg.get("b", seg.get("end"))
            if a is None or b is None:
                continue
            p0 = _world_point(ctx, a)
            p1 = _world_point(ctx, b)
            if not scene_pts:
                scene_pts.append(p0)
            else:
                last = scene_pts[-1]
                if np.linalg.norm(last[:2] - p0[:2]) > 1e-6:
                    scene_pts.append(p0)
            scene_pts.append(p1)
            if np.linalg.norm(p1[:2] - p0[:2]) > 1e-6:
                track_segments.append({"type": "line", "p0": p0, "p1": p1})
        elif stype == "arc":
            center = seg.get("center", (0.0, 0.0))
            radius = float(seg.get("radius", 1.0))
            start_deg = float(seg.get("start_angle_deg", 0.0))
            angle_deg = seg.get("angle_deg")
            end_deg = seg.get("end_angle_deg")
            if angle_deg is None and end_deg is None:
                angle_deg = 90.0
            if end_deg is None:
                end_deg = start_deg + float(angle_deg)
            samples = int(seg.get("samples", 24))
            samples = max(2, samples)
            angles = np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), samples)
            cx, cy = float(center[0]), float(center[1])
            pts = []
            for ang in angles:
                x = cx + radius * np.cos(ang)
                y = cy + radius * np.sin(ang)
                pts.append(_world_point(ctx, (x, y)))
            if pts:
                if not scene_pts:
                    scene_pts.append(pts[0])
                else:
                    last = scene_pts[-1]
                    if np.linalg.norm(last[:2] - pts[0][:2]) > 1e-6:
                        scene_pts.append(pts[0])
                scene_pts.extend(pts[1:])
                if scale_is_uniform:
                    center_scene = _world_point(ctx, (cx, cy))
                    radius_scene = radius * min(ctx.scale_x, ctx.scale_y)
                    sweep = np.deg2rad(end_deg - start_deg)
                    track_segments.append(
                        {
                            "type": "arc",
                            "center": center_scene,
                            "radius": radius_scene,
                            "start": np.deg2rad(start_deg),
                            "sweep": sweep,
                            "sign": 1.0 if sweep >= 0 else -1.0,
                        }
                    )
                else:
                    for idx in range(len(pts) - 1):
                        p0 = pts[idx]
                        p1 = pts[idx + 1]
                        if np.linalg.norm(p1[:2] - p0[:2]) > 1e-6:
                            track_segments.append({"type": "line", "p0": p0, "p1": p1})
        elif stype == "polyline":
            pts = seg.get("points") or []
            prev = None
            for pt in pts:
                p = _world_point(ctx, pt)
                if not scene_pts:
                    scene_pts.append(p)
                else:
                    last = scene_pts[-1]
                    if np.linalg.norm(last[:2] - p[:2]) > 1e-6:
                        scene_pts.append(p)
                if prev is not None and np.linalg.norm(p[:2] - prev[:2]) > 1e-6:
                    track_segments.append({"type": "line", "p0": prev, "p1": p})
                prev = p
    return scene_pts, track_segments


def _build_track_data(
    scene_pts: List[np.ndarray],
    *,
    normal_mode: str,
    track_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    # remove consecutive duplicates
    cleaned: List[np.ndarray] = []
    for pt in scene_pts:
        if not cleaned:
            cleaned.append(pt)
            continue
        if np.linalg.norm(cleaned[-1][:2] - pt[:2]) > 1e-6:
            cleaned.append(pt)
    if len(cleaned) < 2:
        cleaned = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]

    segments = []
    total = 0.0
    cum = [0.0]
    if track_segments:
        for seg in track_segments:
            stype = str(seg.get("type", "line")).strip().lower()
            if stype == "line":
                p0 = seg.get("p0")
                p1 = seg.get("p1")
                if p0 is None or p1 is None:
                    continue
                vec = p1 - p0
                seg_len = float(np.linalg.norm(vec[:2]))
                if seg_len < 1e-6:
                    continue
                tangent = vec / seg_len
                normal = np.array([-tangent[1], tangent[0], 0.0], dtype=float)
                normal = _normalize_normal(normal, normal_mode)
                segments.append(
                    {"type": "line", "p0": p0, "p1": p1, "len": seg_len, "tan": tangent, "norm": normal}
                )
                total += seg_len
                cum.append(total)
            elif stype == "arc":
                center = seg.get("center")
                radius = float(seg.get("radius", 0.0))
                start = float(seg.get("start", 0.0))
                sweep = float(seg.get("sweep", 0.0))
                if center is None or radius <= 1e-6 or abs(sweep) <= 1e-6:
                    continue
                seg_len = abs(sweep) * radius
                segments.append(
                    {
                        "type": "arc",
                        "center": center,
                        "radius": radius,
                        "start": start,
                        "sweep": sweep,
                        "sign": 1.0 if sweep >= 0 else -1.0,
                        "len": seg_len,
                    }
                )
                total += seg_len
                cum.append(total)
    else:
        for i in range(len(cleaned) - 1):
            p0 = cleaned[i]
            p1 = cleaned[i + 1]
            vec = p1 - p0
            seg_len = float(np.linalg.norm(vec[:2]))
            if seg_len < 1e-6:
                continue
            tangent = vec / seg_len
            normal = np.array([-tangent[1], tangent[0], 0.0], dtype=float)
            normal = _normalize_normal(normal, normal_mode)
            segments.append({"type": "line", "p0": p0, "p1": p1, "len": seg_len, "tan": tangent, "norm": normal})
            total += seg_len
            cum.append(total)
    if not segments:
        segments = [{"p0": cleaned[0], "p1": cleaned[1], "len": 1.0, "tan": np.array([1.0, 0.0, 0.0]), "norm": np.array([0.0, 1.0, 0.0])}]
        total = 1.0
        cum = [0.0, 1.0]
    return {"segments": segments, "cum": cum, "total": total, "normal_mode": normal_mode}


@register("track")
def track(*, spec: Dict[str, Any], ctx: World2DContext, theme, **_) -> BuildResult:
    scene_pts, track_segments = _build_track_points(spec, ctx)
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

    normal_mode = str(spec.get("normal_mode", "up")).strip().lower()
    data = _build_track_data(scene_pts, normal_mode=normal_mode, track_segments=track_segments)
    setattr(group, "_track_data", data)
    return BuildResult(group, id_map=_register_id(spec, group))
