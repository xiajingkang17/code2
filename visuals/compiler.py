from __future__ import annotations

from math import atan2, cos, radians, sin, floor, log10
from typing import Any, Dict, List, Optional, Tuple
import re


_DEFAULT_X_RANGE = (-6.0, 6.0)
_DEFAULT_Y_RANGE = (-3.5, 3.5)
_DEFAULT_PLANE_LENGTH = 4.5
_DEFAULT_BASE = (-4.5, -2.4)
_DEFAULT_FORCE_LEN = 1.1
_BLOCK_WIDTH = 0.8
_BLOCK_HEIGHT = 0.5
_BLOCK_MARGIN = 0.0
_FORCE_FONT_SIZE = 14.0
_RANGE_PADDING = 0.6
_MIN_SPAN_RATIO = 0.45
_LABEL_OFFSET_RATIO = 0.18
_LABEL_BACK_RATIO = 0.2
_SURFACE_COLOR = "#ffffff"
_BLOCK_COLOR = "#ffffff"
_FORCE_LABEL_MAX = 16.0
_HORIZ_RATIO_CAP = 3.0
_RANGE_OVERSIZE = 1.3
_SNAP_THRESHOLD = 1.1
_SCALE_KEYS = {
    "width",
    "height",
    "radius",
    "length",
    "size",
    "thickness",
    "corner_radius",
    "tip_length",
    "edge_height",
    "block_offset",
    "block_w",
    "block_h",
    "entry_length",
    "exit_length",
    "amplitude",
    "dash",
    "gap",
    "hatch_length",
    "top_hatch_length",
    "edge_hatch_length",
    "line_hatch_length",
    "r",
}


def compile_plan_visuals(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    questions = plan_dict.get("questions", [])
    if not isinstance(questions, list):
        return plan_dict
    for q in questions:
        if not isinstance(q, dict):
            continue
        visual = q.get("visual")
        if isinstance(visual, dict):
            diagram_spec = q.get("diagram_spec")
            if diagram_spec and "diagram_spec" not in visual:
                visual = dict(visual)
                visual["diagram_spec"] = diagram_spec
                q["visual"] = visual
            q["visual"] = compile_visual_spec(visual)
    return plan_dict


def compile_visual_spec(spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(spec, dict):
        return spec
    ctype = str(spec.get("type", "")).strip().lower()
    diagram_spec = spec.get("diagram_spec")
    if ctype in {"world2d", "world3d", "group"}:
        if ctype == "world2d":
            return _normalize_world2d(spec)
        return spec
    if "objects" in spec and (not ctype or ctype == "objects"):
        compiled = _compile_objects(spec)
        if diagram_spec and isinstance(compiled, dict) and "diagram_spec" not in compiled:
            compiled["diagram_spec"] = diagram_spec
        return compiled
    if ctype == "inclined_plane" and _looks_like_high_level_incline(spec):
        compiled = _compile_inclined_plane(spec)
        if diagram_spec and isinstance(compiled, dict) and "diagram_spec" not in compiled:
            compiled["diagram_spec"] = diagram_spec
        return compiled
    return spec


def _looks_like_high_level_incline(spec: Dict[str, Any]) -> bool:
    if "forces" in spec or "block_pos" in spec or "length_scale" in spec:
        return True
    if "angle" in spec and "angle_deg" not in spec and "base" not in spec and "length" not in spec:
        return True
    return False


def _compile_inclined_plane(spec: Dict[str, Any]) -> Dict[str, Any]:
    used_ids: set[str] = set()
    angle_deg = _to_float(spec.get("angle_deg", spec.get("angle", 30.0)), 30.0)
    length_scale = _to_float(spec.get("length_scale", 1.0), 1.0)
    length_scale = _clamp(length_scale, 0.4, 1.6)
    length = _normalize_length(spec.get("length"), length_scale)

    base = _to_point(spec.get("base"), _DEFAULT_BASE)
    slope = _unit_from_angle(angle_deg)
    normal = (-slope[1], slope[0])

    block_w = _to_float(spec.get("block_width", spec.get("block_w", _BLOCK_WIDTH)), _BLOCK_WIDTH)
    block_h = _to_float(spec.get("block_height", spec.get("block_h", _BLOCK_HEIGHT)), _BLOCK_HEIGHT)
    block_pos = _to_float(spec.get("block_pos", 0.55), 0.55)
    block_pos = _clamp(block_pos, 0.05, 0.95)
    raw_offset = spec.get("block_offset")
    if raw_offset is None:
        block_offset = block_h * 0.5 + _BLOCK_MARGIN
    else:
        block_offset = _to_float(raw_offset, block_h * 0.5 + _BLOCK_MARGIN)
    block_center = (
        base[0] + slope[0] * length * block_pos + normal[0] * block_offset,
        base[1] + slope[1] * length * block_pos + normal[1] * block_offset,
    )

    plane_id = _unique_id(_safe_id(spec.get("id") or spec.get("plane_id") or "slope"), used_ids)
    block_id = _unique_id(_safe_id(spec.get("block_id") or "block"), used_ids)
    path_id = _unique_id(_safe_id(spec.get("path_id") or f"path_{plane_id}"), used_ids)

    children: List[Dict[str, Any]] = []
    bounds = _init_bounds()
    children.append(
        {
            "type": "inclined_plane",
            "start": list(base),
            "end": [base[0] + slope[0] * length, base[1] + slope[1] * length],
            "angle_deg": angle_deg,
            "id": plane_id,
        }
    )
    _update_bounds(bounds, base[0], base[1])
    _update_bounds(bounds, base[0] + slope[0] * length, base[1] + slope[1] * length)
    children.append(
        {
            "type": "path_polyline",
            "points": [
                [base[0], base[1]],
                [base[0] + slope[0] * length, base[1] + slope[1] * length],
            ],
            "id": path_id,
            "visible": False,
        }
    )
    block_shape_id = _unique_id(_safe_id(f"{block_id}_shape"), used_ids)
    block_child = {
        "type": "block",
        "pos": [block_center[0], block_center[1]],
        "width": block_w,
        "height": block_h,
        "angle_deg": angle_deg,
        "fill_opacity": 0.0,
        "id": block_shape_id,
    }
    _update_bounds_box(bounds, block_center[0], block_center[1], block_w, block_h, angle_deg)

    forces = spec.get("forces", []) or []
    force_children: List[Dict[str, Any]] = []
    if isinstance(forces, list):
        for idx, f in enumerate(forces, start=1):
            if not isinstance(f, dict):
                continue
            direction = str(f.get("dir", "down")).strip().lower()
            label = str(f.get("label", "")).strip()
            vec = _direction_vector(direction, slope, normal)
            if vec is None:
                continue
            start = block_center
            end = (start[0] + vec[0] * _DEFAULT_FORCE_LEN, start[1] + vec[1] * _DEFAULT_FORCE_LEN)
            force_id = _safe_id(label) if label else f"force_{idx}"
            force_id = _unique_id(force_id, used_ids)
            force_children.append(
                {
                    "type": "force_arrow",
                    "start": [start[0], start[1]],
                    "end": [end[0], end[1]],
                    "label": label,
                    "font_size": _FORCE_FONT_SIZE,
                    "label_pos": "tip",
                    "label_offset_ratio": _LABEL_OFFSET_RATIO,
                    "label_back_ratio": _LABEL_BACK_RATIO,
                    "id": force_id,
                }
            )
            _update_bounds(bounds, end[0], end[1])

    if force_children:
        children.append(
            {
                "type": "group",
                "id": block_id,
                "children": [block_child, *force_children],
            }
        )
    else:
        block_child["id"] = block_id
        children.append(block_child)

    x_range, y_range = _auto_ranges(bounds)
    return {
        "type": "world2d",
        "x_range": list(x_range),
        "y_range": list(y_range),
        "children": children,
    }


def _compile_objects(spec: Dict[str, Any]) -> Dict[str, Any]:
    objects = spec.get("objects", [])
    if not isinstance(objects, list):
        return spec
    used_ids: set[str] = set()
    plane_obj = None
    for obj in objects:
        if isinstance(obj, dict) and str(obj.get("type", "")).strip().lower() == "plane":
            plane_obj = obj
            break

    base = _to_point(spec.get("base") or (plane_obj or {}).get("base"), _DEFAULT_BASE)
    angle_deg = _to_float((plane_obj or {}).get("angle", (plane_obj or {}).get("angle_deg", 0.0)), 0.0)
    length = _normalize_length((plane_obj or {}).get("length"), 1.0)
    slope = _unit_from_angle(angle_deg)
    normal = (-slope[1], slope[0])
    start = (plane_obj or {}).get("start")
    end = (plane_obj or {}).get("end")
    if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
        p1 = (float(start[0]), float(start[1]))
        p2 = (float(end[0]), float(end[1]))
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        if length > 1e-6:
            slope = (dx / length, dy / length)
            normal = (-slope[1], slope[0])
    else:
        p1 = base
        p2 = (base[0] + slope[0] * length, base[1] + slope[1] * length)

    children: List[Dict[str, Any]] = []
    bounds = _init_bounds()
    if plane_obj is not None:
        plane_id = _unique_id(_safe_id(plane_obj.get("id") or "plane"), used_ids)
        plane_type = "rough_surface" if abs(angle_deg) <= 5 else "inclined_plane"
        children.append(
            {
                "type": plane_type,
                "start": [p1[0], p1[1]],
                "end": [p2[0], p2[1]],
                "angle_deg": angle_deg,
                "id": plane_id,
            }
        )
        _update_bounds(bounds, p1[0], p1[1])
        _update_bounds(bounds, p2[0], p2[1])
        path_id = _unique_id(_safe_id(f"path_{plane_id}"), used_ids)
        children.append(
            {
                "type": "path_polyline",
                "points": [
                    [p1[0], p1[1]],
                    [p2[0], p2[1]],
                ],
                "id": path_id,
                "visible": False,
            }
        )

    block_center = None
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        otype = str(obj.get("type", "")).strip().lower()
        if otype != "block":
            continue
        pos = obj.get("pos", 0.5)
        block_w = _to_float(obj.get("width", _BLOCK_WIDTH), _BLOCK_WIDTH)
        block_h = _to_float(obj.get("height", _BLOCK_HEIGHT), _BLOCK_HEIGHT)
        raw_offset = obj.get("block_offset")
        if raw_offset is None:
            block_offset = block_h * 0.5 + _BLOCK_MARGIN
        else:
            block_offset = _to_float(raw_offset, block_h * 0.5 + _BLOCK_MARGIN)
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            bx, by = float(pos[0]), float(pos[1])
        elif plane_obj is not None:
            t = _clamp(_to_float(pos, 0.5), 0.0, 1.0)
            bx = p1[0] + slope[0] * length * t + normal[0] * block_offset
            by = p1[1] + slope[1] * length * t + normal[1] * block_offset
        else:
            bx, by = 0.0, 0.0
        block_center = (bx, by)
        block_id = _unique_id(_safe_id(obj.get("id") or "block"), used_ids)
        block_shape_id = _unique_id(_safe_id(f"{block_id}_shape"), used_ids)
        block_child = {
            "type": "block",
            "pos": [bx, by],
            "width": block_w,
            "height": block_h,
            "angle_deg": angle_deg,
            "fill_opacity": 0.0,
            "id": block_shape_id,
        }
        _update_bounds_box(bounds, bx, by, block_w, block_h, angle_deg)

    force_children = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        otype = str(obj.get("type", "")).strip().lower()
        if otype != "force":
            continue
        direction = str(obj.get("dir", "right")).strip().lower()
        label = str(obj.get("label", "")).strip()
        vec = _direction_vector(direction, slope, normal)
        if vec is None:
            continue
        if block_center is None:
            block_center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + (_BLOCK_HEIGHT * 0.5 + _BLOCK_MARGIN))
        start = block_center
        end = (start[0] + vec[0] * _DEFAULT_FORCE_LEN, start[1] + vec[1] * _DEFAULT_FORCE_LEN)
        force_id = _safe_id(label) if label else f"force_{len(used_ids) + 1}"
        force_id = _unique_id(force_id, used_ids)
        force_children.append(
            {
                "type": "force_arrow",
                "start": [start[0], start[1]],
                "end": [end[0], end[1]],
                "label": label,
                "font_size": _FORCE_FONT_SIZE,
                "label_pos": "tip",
                "label_offset_ratio": _LABEL_OFFSET_RATIO,
                "label_back_ratio": _LABEL_BACK_RATIO,
                "id": force_id,
            }
        )
        _update_bounds(bounds, end[0], end[1])

    if block_center is not None:
        if force_children:
            children.append(
                {
                    "type": "group",
                    "id": block_id,
                    "children": [block_child, *force_children],
                }
            )
        else:
            block_child["id"] = block_id
            children.append(block_child)

    x_range, y_range = _auto_ranges(bounds)
    return {
        "type": "world2d",
        "x_range": list(x_range),
        "y_range": list(y_range),
        "children": children,
    }


def _normalize_world2d(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(spec)
    raw_children = list(out.get("children", []) or [])
    diagram_spec = _parse_diagram_spec(out.get("diagram_spec")) or {}
    has_diagram_spec = bool(diagram_spec)
    origin_val = diagram_spec.get("origin", (0.0, 0.0))
    try:
        origin = (float(origin_val[0]), float(origin_val[1])) if isinstance(origin_val, (list, tuple)) else (0.0, 0.0)
    except Exception:
        origin = (0.0, 0.0)

    explicit_x = isinstance(diagram_spec.get("x_range"), (list, tuple)) and len(diagram_spec.get("x_range")) >= 2
    explicit_y = isinstance(diagram_spec.get("y_range"), (list, tuple)) and len(diagram_spec.get("y_range")) >= 2
    has_out_x = isinstance(out.get("x_range"), (list, tuple)) and len(out.get("x_range")) >= 2
    has_out_y = isinstance(out.get("y_range"), (list, tuple)) and len(out.get("y_range")) >= 2
    lock_range = bool(diagram_spec.get("lock_range") or diagram_spec.get("fixed_range"))
    if explicit_x or explicit_y or has_out_x or has_out_y:
        lock_range = True

    target_x = diagram_spec.get("x_range") if explicit_x else out.get("x_range")
    target_y = diagram_spec.get("y_range") if explicit_y else out.get("y_range")
    if not (isinstance(target_x, (list, tuple)) and len(target_x) >= 2):
        target_x = _DEFAULT_X_RANGE
    if not (isinstance(target_y, (list, tuple)) and len(target_y) >= 2):
        target_y = _DEFAULT_Y_RANGE
    target_x = (float(target_x[0]), float(target_x[1]))
    target_y = (float(target_y[0]), float(target_y[1]))

    raw_u = diagram_spec.get("u", diagram_spec.get("unit", diagram_spec.get("scale")))
    apply_scale = bool(diagram_spec.get("apply_scale"))
    if isinstance(raw_u, str) and raw_u.strip().lower() == "auto":
        raw_u = None
    u_val = None
    if raw_u is not None:
        try:
            u_val = float(raw_u)
        except Exception:
            u_val = None

    bounds = _init_bounds()
    for node in raw_children:
        if isinstance(node, dict):
            _collect_bounds_from_node(node, bounds)
    has_bounds = bounds[0] is not None

    if apply_scale and u_val is None and has_bounds:
        u_val = _compute_auto_u(raw_children, target_x, target_y)
    if apply_scale and u_val is None:
        apply_scale = False

    if apply_scale and u_val is not None and u_val > 0:
        quantize = diagram_spec.get("quantize", True)
        if quantize:
            u_val = _quantize_u(u_val)
        scale = 1.0 / u_val
        for child in raw_children:
            if isinstance(child, dict):
                _scale_node(child, scale, origin)
        if isinstance(out.get("x_range"), (list, tuple)) and "x_range" not in diagram_spec:
            try:
                out["x_range"] = [
                    origin[0] + (float(out["x_range"][0]) - origin[0]) * scale,
                    origin[0] + (float(out["x_range"][1]) - origin[0]) * scale,
                ]
            except Exception:
                pass
        if isinstance(out.get("y_range"), (list, tuple)) and "y_range" not in diagram_spec:
            try:
                out["y_range"] = [
                    origin[1] + (float(out["y_range"][0]) - origin[1]) * scale,
                    origin[1] + (float(out["y_range"][1]) - origin[1]) * scale,
                ]
            except Exception:
                pass
    if explicit_x:
        out["x_range"] = list(target_x)
    if explicit_y:
        out["y_range"] = list(target_y)
    out["_lock_range"] = lock_range

    children = []
    for child in raw_children:
        if not isinstance(child, dict):
            continue
        node = _normalize_node(child)
        if node is not None:
            children.append(node)

    children = _compress_connected_horizontal(children)
    debug_mode = bool(diagram_spec.get("debug", False)) if diagram_spec else bool(out.get("debug", False))
    children = _snap_blocks_to_planes(children, debug=debug_mode)
    out["children"] = children
    if not out.get("_lock_range"):
        _maybe_autofit_ranges(out, children)
    out.pop("_lock_range", None)
    return out


def _normalize_node(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    node = dict(node)
    ctype = str(node.get("type", "")).strip().lower()
    node_id = str(node.get("id", "")).strip().lower()
    style = node.pop("style", None)
    if isinstance(style, dict):
        if "color" in style and "color" not in node:
            node["color"] = style["color"]
        if "stroke" in style and "color" not in node:
            node["color"] = style["stroke"]
        if "width" in style and "stroke_width" not in node:
            node["stroke_width"] = style["width"]
        if "font_size" in style and "font_size" not in node:
            node["font_size"] = style["font_size"]
        if "label" in style and "label" not in node:
            node["label"] = style["label"]
        if "label_offset" in style and "label_offset" not in node:
            node["label_offset"] = style["label_offset"]
    if "angle" in node and "angle_deg" not in node:
        node["angle_deg"] = node["angle"]

    if ctype in {"line", "arrow", "force_arrow"}:
        if "start" not in node and "p1" in node:
            node["start"] = node.get("p1")
        if "end" not in node and "p2" in node:
            node["end"] = node.get("p2")
        if ctype == "line":
            inferred = _infer_surface_type(node, node_id)
            if inferred:
                node["type"] = inferred
                ctype = inferred

    if ctype == "circle":
        if "center" not in node and "pos" in node:
            node["center"] = node.get("pos")
        if "radius" not in node and "r" in node:
            node["radius"] = node.get("r")
        if _looks_like_block_id(node_id):
            node["type"] = "block"
            ctype = "block"
            if "pos" not in node:
                node["pos"] = node.get("center", node.get("pos"))
            if "width" not in node and "height" not in node:
                try:
                    radius = float(node.get("radius", node.get("r", _BLOCK_WIDTH * 0.5)))
                    node["width"] = radius * 2.0
                    node["height"] = radius * 2.0
                except Exception:
                    pass

    if ctype in {"inclined_plane", "rough_surface", "smooth_surface", "horizontal_surface"}:
        if "start" not in node and "end" not in node:
            pts = node.get("points")
            if isinstance(pts, (list, tuple)) and len(pts) >= 2:
                node["start"] = pts[0]
                node["end"] = pts[-1]

    if ctype == "text":
        text = str(node.get("text", "")).strip().lower()
        cid = str(node.get("id", "")).strip().lower()
        if cid.startswith("label_") or text.startswith(("start", "bottom", "stop")):
            return None

    if ctype == "group":
        children = []
        for child in node.get("children", []) or []:
            if not isinstance(child, dict):
                continue
            normalized = _normalize_node(child)
            if normalized is not None:
                children.append(normalized)
        node["children"] = children
        return node

    if ctype in {"inclined_plane", "rough_surface", "smooth_surface", "horizontal_surface"}:
        node["color"] = _normalize_color(node.get("color"), _SURFACE_COLOR)
        if "stroke_width" not in node:
            node["stroke_width"] = 2.6
    elif ctype == "block":
        if "fill_opacity" not in node:
            node["fill_opacity"] = 0.0
        node["color"] = _normalize_color(node.get("color"), _BLOCK_COLOR)
        width = _to_float(node.get("width", _BLOCK_WIDTH), _BLOCK_WIDTH)
        height = _to_float(node.get("height", _BLOCK_HEIGHT), _BLOCK_HEIGHT)
        size = max(width, height)
        node["width"] = size
        node["height"] = size
        if "stroke_width" not in node:
            node["stroke_width"] = 2.8
    elif ctype == "force_arrow":
        node.setdefault("label_pos", "tip")
        node.setdefault("label_offset_ratio", _LABEL_OFFSET_RATIO)
        node.setdefault("label_back_ratio", _LABEL_BACK_RATIO)
        if "font_size" not in node:
            node["font_size"] = _FORCE_FONT_SIZE
        else:
            try:
                node["font_size"] = min(float(node["font_size"]), _FORCE_LABEL_MAX)
            except Exception:
                node["font_size"] = _FORCE_FONT_SIZE
    return node


def _parse_diagram_spec(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        if set(raw.keys()) == {"text"}:
            return _parse_diagram_spec_text(raw.get("text", ""))
        return raw
    if isinstance(raw, str):
        return _parse_diagram_spec_text(raw)
    return None


def _parse_diagram_spec_text(text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    content = text.strip()
    if not content:
        return None
    spec: Dict[str, Any] = {}

    def _parse_range(key: str):
        pattern = rf"{key}\s*:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]"
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            try:
                spec[key] = [float(match.group(1)), float(match.group(2))]
            except Exception:
                pass

    _parse_range("x_range")
    _parse_range("y_range")

    origin_match = re.search(r"origin\s*:\s*[\[(]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\])]", content, flags=re.IGNORECASE)
    if origin_match:
        try:
            spec["origin"] = [float(origin_match.group(1)), float(origin_match.group(2))]
        except Exception:
            pass

    u_match = re.search(r"\bu\b\s*:\s*([^\n]+)", content, flags=re.IGNORECASE)
    if u_match:
        u_text = u_match.group(1).strip().lower()
        if "auto" in u_text:
            spec["u"] = "auto"
        else:
            num_match = re.search(r"[-+]?\d*\.?\d+", u_text)
            if num_match:
                try:
                    spec["u"] = float(num_match.group(0))
                except Exception:
                    pass

    if re.search(r"\block_range\b|\bfixed_range\b", content, flags=re.IGNORECASE):
        spec["lock_range"] = True

    return spec or None


def _quantize_u(u: float) -> float:
    if u <= 0:
        return u
    exp = int(floor(log10(u)))
    base = u / (10 ** exp)
    candidates = [1.0, 2.0, 5.0, 10.0]
    best = min(candidates, key=lambda c: abs(c - base))
    return best * (10 ** exp)


def _update_bounds_from_point(bounds: List[Optional[float]], pt: Any) -> None:
    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
        try:
            _update_bounds(bounds, float(pt[0]), float(pt[1]))
        except Exception:
            pass


def _collect_bounds_from_node(node: Dict[str, Any], bounds: List[Optional[float]]) -> None:
    if not isinstance(node, dict):
        return
    for key in ("pos", "start", "end", "center", "a", "b", "origin", "p1", "p2"):
        _update_bounds_from_point(bounds, node.get(key))
    pts = node.get("points")
    if isinstance(pts, (list, tuple)):
        for pt in pts:
            _update_bounds_from_point(bounds, pt)
    segs = node.get("segments")
    if isinstance(segs, list):
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            for key in ("a", "b", "start", "end", "center"):
                _update_bounds_from_point(bounds, seg.get(key))
            pts = seg.get("points")
            if isinstance(pts, (list, tuple)):
                for pt in pts:
                    _update_bounds_from_point(bounds, pt)
            if isinstance(seg.get("center"), (list, tuple)) and "radius" in seg:
                try:
                    cx, cy = float(seg["center"][0]), float(seg["center"][1])
                    r = float(seg["radius"])
                    _update_bounds(bounds, cx - r, cy - r)
                    _update_bounds(bounds, cx + r, cy + r)
                except Exception:
                    pass
    center = node.get("center") or node.get("pos")
    radius = node.get("radius", node.get("r"))
    if isinstance(center, (list, tuple)) and len(center) >= 2 and radius is not None:
        try:
            cx, cy = float(center[0]), float(center[1])
            r = float(radius)
            _update_bounds(bounds, cx - r, cy - r)
            _update_bounds(bounds, cx + r, cy + r)
        except Exception:
            pass
    if "children" in node and isinstance(node.get("children"), list):
        for child in node["children"]:
            if isinstance(child, dict):
                _collect_bounds_from_node(child, bounds)


def _compute_auto_u(
    children: List[Dict[str, Any]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Optional[float]:
    bounds = _init_bounds()
    for node in children:
        if isinstance(node, dict):
            _collect_bounds_from_node(node, bounds)
    if bounds[0] is None:
        return None
    min_x, max_x, min_y, max_y = bounds  # type: ignore[misc]
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    target_span_x = max(1e-6, x_range[1] - x_range[0])
    target_span_y = max(1e-6, y_range[1] - y_range[0])
    scale = min(target_span_x / span_x, target_span_y / span_y)
    if scale <= 1e-9:
        return None
    return 1.0 / scale


def _collect_planes(children: List[Dict[str, Any]]) -> List[Tuple[float, float, float, float, float, float, float]]:
    planes = []
    for node in children:
        ctype = str(node.get("type", "")).strip().lower()
        if ctype not in {"inclined_plane", "rough_surface", "smooth_surface", "horizontal_surface"}:
            continue
        start = node.get("start")
        end = node.get("end")
        if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2):
            continue
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        dx = ex - sx
        dy = ey - sy
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len <= 1e-6:
            continue
        planes.append((sx, sy, ex, ey, dx / seg_len, dy / seg_len, seg_len))
    return planes


def _nearest_plane(px: float, py: float, planes: List[Tuple[float, float, float, float, float, float, float]]):
    best = None
    for sx, sy, ex, ey, ux, uy, seg_len in planes:
        vx = px - sx
        vy = py - sy
        t = vx * ux + vy * uy
        if seg_len > 1e-6:
            t = max(0.0, min(seg_len, t))
        proj_x = sx + ux * t
        proj_y = sy + uy * t
        nx, ny = -uy, ux
        if ny < 0:
            nx, ny = -nx, -ny
        dist = (px - proj_x) * nx + (py - proj_y) * ny
        dist_abs = abs(dist)
        best = (dist_abs, proj_x, proj_y, nx, ny, ux, uy) if best is None or dist_abs < best[0] else best
    return best


def _angle_locked(node: Dict[str, Any]) -> bool:
    if node.get("angle_lock") is True:
        return True
    if node.get("angle_auto") is False:
        return True
    return False


def _translate_node(node: Dict[str, Any], dx: float, dy: float) -> None:
    if not isinstance(node, dict):
        return
    for key in ("pos", "start", "end", "center"):
        val = node.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                node[key] = [float(val[0]) + dx, float(val[1]) + dy]
            except Exception:
                pass
    pts = node.get("points")
    if isinstance(pts, (list, tuple)):
        new_pts = []
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    new_pts.append([float(pt[0]) + dx, float(pt[1]) + dy])
                except Exception:
                    new_pts.append(list(pt))
            else:
                new_pts.append(pt)
        node["points"] = new_pts
    if "children" in node and isinstance(node.get("children"), list):
        for child in node["children"]:
            if isinstance(child, dict):
                _translate_node(child, dx, dy)


def _scale_point(pt: Tuple[float, float], scale: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    ox, oy = origin
    return ox + (pt[0] - ox) * scale, oy + (pt[1] - oy) * scale


def _scale_node(node: Dict[str, Any], scale: float, origin: Tuple[float, float]) -> None:
    if not isinstance(node, dict):
        return
    for key in ("pos", "start", "end", "center", "a", "b", "origin", "p1", "p2"):
        val = node.get(key)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                x, y = float(val[0]), float(val[1])
                sx, sy = _scale_point((x, y), scale, origin)
                node[key] = [sx, sy]
            except Exception:
                pass
    pts = node.get("points")
    if isinstance(pts, (list, tuple)):
        new_pts = []
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    x, y = float(pt[0]), float(pt[1])
                    sx, sy = _scale_point((x, y), scale, origin)
                    new_pts.append([sx, sy])
                except Exception:
                    new_pts.append(list(pt))
            else:
                new_pts.append(pt)
        node["points"] = new_pts
    segs = node.get("segments")
    if isinstance(segs, list):
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            for key in ("a", "b", "start", "end", "center"):
                val = seg.get(key)
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    try:
                        x, y = float(val[0]), float(val[1])
                        sx, sy = _scale_point((x, y), scale, origin)
                        seg[key] = [sx, sy]
                    except Exception:
                        pass
            if "radius" in seg:
                try:
                    seg["radius"] = float(seg["radius"]) * scale
                except Exception:
                    pass
            pts = seg.get("points")
            if isinstance(pts, (list, tuple)):
                new_pts = []
                for pt in pts:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        try:
                            x, y = float(pt[0]), float(pt[1])
                            sx, sy = _scale_point((x, y), scale, origin)
                            new_pts.append([sx, sy])
                        except Exception:
                            new_pts.append(list(pt))
                    else:
                        new_pts.append(pt)
                seg["points"] = new_pts
    for key in list(node.keys()):
        if key in _SCALE_KEYS:
            val = node.get(key)
            if isinstance(val, (int, float)):
                node[key] = float(val) * scale
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    node[key] = [float(val[0]) * scale, float(val[1]) * scale]
                except Exception:
                    pass
    if "children" in node and isinstance(node.get("children"), list):
        for child in node["children"]:
            if isinstance(child, dict):
                _scale_node(child, scale, origin)


def _looks_like_block_id(node_id: str) -> bool:
    if not node_id:
        return False
    if "block" in node_id and "ball" not in node_id:
        return True
    return False


def _infer_surface_type(node: Dict[str, Any], node_id: str) -> Optional[str]:
    if not node_id or "path" in node_id or node_id.startswith("label_"):
        return None
    if "inclined" in node_id or "slope" in node_id:
        return "inclined_plane"
    if "rough" in node_id:
        return "rough_surface"
    if "smooth" in node_id:
        return "smooth_surface"
    if "horizontal" in node_id:
        return "horizontal_surface"
    if "ground" in node_id or "surface" in node_id or "floor" in node_id:
        start = node.get("start")
        end = node.get("end")
        try:
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
                if abs(float(start[1]) - float(end[1])) <= 1e-3:
                    return "horizontal_surface"
        except Exception:
            pass
        return "rough_surface"
    return None


def _find_child_by_id(children: List[Dict[str, Any]], obj_id: str) -> Optional[Dict[str, Any]]:
    for child in children:
        if not isinstance(child, dict):
            continue
        if str(child.get("id", "")).strip() == obj_id:
            return child
    return None


def _snap_blocks_to_planes(children: List[Dict[str, Any]], *, debug: bool = False) -> List[Dict[str, Any]]:
    planes = _collect_planes(children)
    if not planes:
        return children

    adjusted = []
    for node in children:
        ctype = str(node.get("type", "")).strip().lower()
        if ctype == "block":
            adjusted.append(_wrap_block_node(node, planes, debug=debug))
        elif ctype == "group":
            adjusted.append(_snap_group_node(node, planes, debug=debug))
        else:
            adjusted.append(node)
    return adjusted


def _wrap_block_node(node: Dict[str, Any], planes: List[Tuple[float, float, float, float, float, float, float]], *, debug: bool = False) -> Dict[str, Any]:
    if node.get("snap_disable"):
        return node
    pos = node.get("pos")
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        return node
    px, py = float(pos[0]), float(pos[1])
    height = _to_float(node.get("height", _BLOCK_HEIGHT), _BLOCK_HEIGHT)
    best = _nearest_plane(px, py, planes)
    if best is None:
        return node
    dist_abs, proj_x, proj_y, nx, ny, ux, uy = best
    desired = height * 0.5 + _BLOCK_MARGIN
    if dist_abs >= height * _SNAP_THRESHOLD:
        return node

    new_px = proj_x + nx * desired
    new_py = proj_y + ny * desired
    block = dict(node)
    block["pos"] = [new_px, new_py]
    if not _angle_locked(block):
        block["angle_deg"] = atan2(uy, ux) * 180.0 / 3.141592653589793

    block_id = str(block.get("id") or "").strip()
    if not block_id:
        return block
    shape_id = f"{block_id}_shape"
    block["id"] = shape_id

    anchor = {
        "type": "point",
        "id": f"{block_id}_anchor",
        "pos": [proj_x, proj_y],
        "radius": 0.05 if debug else 0.04,
        "visible": True if debug else False,
    }
    return {"type": "group", "id": block_id, "angle_base_deg": block.get("angle_deg"), "children": [block, anchor]}


def _snap_group_node(node: Dict[str, Any], planes: List[Tuple[float, float, float, float, float, float, float]], *, debug: bool = False) -> Dict[str, Any]:
    group = dict(node)
    children = list(group.get("children", []) or [])
    block_child = None
    for child in children:
        if isinstance(child, dict) and str(child.get("type", "")).strip().lower() == "block":
            block_child = child
            break
    if block_child is None:
        group["children"] = children
        return group
    if group.get("snap_disable") or block_child.get("snap_disable"):
        group["children"] = children
        return group

    group_id = str(group.get("id") or "").strip()
    if group_id:
        child_id = str(block_child.get("id") or "").strip()
        if child_id == group_id:
            block_child["id"] = f"{group_id}_shape"

    pos = block_child.get("pos")
    if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
        group["children"] = children
        return group
    px, py = float(pos[0]), float(pos[1])
    height = _to_float(block_child.get("height", _BLOCK_HEIGHT), _BLOCK_HEIGHT)
    best = _nearest_plane(px, py, planes)
    if best is None:
        group["children"] = children
        return group
    dist_abs, proj_x, proj_y, nx, ny, ux, uy = best
    desired = height * 0.5 + _BLOCK_MARGIN
    if dist_abs >= height * _SNAP_THRESHOLD:
        group["children"] = children
        return group

    new_px = proj_x + nx * desired
    new_py = proj_y + ny * desired
    dx = new_px - px
    dy = new_py - py
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        for child in children:
            if isinstance(child, dict):
                _translate_node(child, dx, dy)

    if not _angle_locked(block_child):
        block_child["angle_deg"] = atan2(uy, ux) * 180.0 / 3.141592653589793

    if group_id:
        anchor_id = f"{group_id}_anchor"
        anchor_child = _find_child_by_id(children, anchor_id)
        if anchor_child is None:
            children.append(
                {
                    "type": "point",
                    "id": anchor_id,
                    "pos": [proj_x, proj_y],
                    "radius": 0.05 if debug else 0.04,
                    "visible": True if debug else False,
                }
            )
        else:
            anchor_child["pos"] = [proj_x, proj_y]

    if group_id:
        group["angle_base_deg"] = block_child.get("angle_deg")
    group["children"] = children
    return group


def _maybe_autofit_ranges(spec: Dict[str, Any], children: List[Dict[str, Any]]) -> None:
    bounds = _init_bounds()
    for node in children:
        ctype = str(node.get("type", "")).strip().lower()
        if ctype in {"inclined_plane", "rough_surface", "smooth_surface", "horizontal_surface", "force_arrow"}:
            start = node.get("start")
            end = node.get("end")
            if isinstance(start, (list, tuple)) and len(start) >= 2:
                _update_bounds(bounds, float(start[0]), float(start[1]))
            if isinstance(end, (list, tuple)) and len(end) >= 2:
                _update_bounds(bounds, float(end[0]), float(end[1]))
        elif ctype == "block":
            pos = node.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                width = _to_float(node.get("width", _BLOCK_WIDTH), _BLOCK_WIDTH)
                height = _to_float(node.get("height", _BLOCK_HEIGHT), _BLOCK_HEIGHT)
                _update_bounds_box(bounds, float(pos[0]), float(pos[1]), width, height, _to_float(node.get("angle_deg", 0.0), 0.0))
        elif ctype == "path_polyline":
            pts = node.get("points")
            if isinstance(pts, (list, tuple)):
                for pt in pts:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        _update_bounds(bounds, float(pt[0]), float(pt[1]))
        elif ctype in {"path_arc", "path_circle"}:
            center = node.get("center")
            radius = _to_float(node.get("radius", 1.0), 1.0)
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                cx, cy = float(center[0]), float(center[1])
                _update_bounds(bounds, cx - radius, cy - radius)
                _update_bounds(bounds, cx + radius, cy + radius)
        elif ctype == "text":
            pos = node.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                _update_bounds(bounds, float(pos[0]), float(pos[1]))

    if bounds[0] is None:
        return
    auto_x, auto_y = _auto_ranges(bounds)
    x_range = spec.get("x_range")
    y_range = spec.get("y_range")
    if _range_is_oversized(x_range, auto_x) or _range_is_oversized(y_range, auto_y):
        spec["x_range"] = list(auto_x)
        spec["y_range"] = list(auto_y)


def _range_is_oversized(raw, auto_range: Tuple[float, float]) -> bool:
    if not (isinstance(raw, (list, tuple)) and len(raw) >= 2):
        return True
    try:
        span = float(raw[1]) - float(raw[0])
    except Exception:
        return True
    auto_span = auto_range[1] - auto_range[0]
    if auto_span <= 1e-6:
        return False
    return span > auto_span * _RANGE_OVERSIZE


def _compress_connected_horizontal(children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    incline = None
    for node in children:
        if str(node.get("type", "")).strip().lower() == "inclined_plane":
            start = node.get("start")
            end = node.get("end")
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
                sx, sy = float(start[0]), float(start[1])
                ex, ey = float(end[0]), float(end[1])
                length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                incline = (sx, sy, ex, ey, length)
                break
    if not incline or incline[4] <= 1e-6:
        return children
    _, _, ix, iy, incline_len = incline
    capped = []
    for node in children:
        ctype = str(node.get("type", "")).strip().lower()
        if ctype in {"horizontal_surface", "rough_surface", "smooth_surface"}:
            start = node.get("start")
            end = node.get("end")
            if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) >= 2 and len(end) >= 2:
                sx, sy = float(start[0]), float(start[1])
                ex, ey = float(end[0]), float(end[1])
                if abs(sx - ix) < 0.3 and abs(sy - iy) < 0.3:
                    dx = ex - sx
                    dy = ey - sy
                    length = (dx * dx + dy * dy) ** 0.5
                    cap = incline_len * _HORIZ_RATIO_CAP
                    if length > cap and length > 1e-6:
                        scale = cap / length
                        node["end"] = [sx + dx * scale, sy + dy * scale]
        capped.append(node)
    return capped


def _normalize_color(value: Any, fallback: str) -> str:
    if not value:
        return fallback
    color = str(value).strip().lower()
    if color in {"#000", "#000000", "black"}:
        return fallback
    if color.startswith("#") and len(color) in {4, 7}:
        try:
            if len(color) == 4:
                r = int(color[1] * 2, 16)
                g = int(color[2] * 2, 16)
                b = int(color[3] * 2, 16)
            else:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            if brightness < 0.35:
                return fallback
        except Exception:
            return fallback
    return color


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_point(value: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return fallback
    return fallback


def _normalize_length(raw: Any, scale: float) -> float:
    if raw is None:
        return _DEFAULT_PLANE_LENGTH * scale
    length = _to_float(raw, _DEFAULT_PLANE_LENGTH * scale)
    if length <= 2.0:
        return _DEFAULT_PLANE_LENGTH * length
    return length


def _unit_from_angle(angle_deg: float) -> Tuple[float, float]:
    angle = radians(angle_deg)
    return cos(angle), sin(angle)


def _direction_vector(
    direction: str, slope: Tuple[float, float], normal: Tuple[float, float]
) -> Optional[Tuple[float, float]]:
    if direction in {"up", "down", "left", "right"}:
        mapping = {"up": (0.0, 1.0), "down": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}
        return mapping[direction]
    if direction == "up_slope":
        return slope
    if direction == "down_slope":
        return (-slope[0], -slope[1])
    if direction == "normal":
        return normal
    return None


def _safe_id(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"^_+", "", text)
    if not text or not text[0].isalpha():
        text = f"id_{text}" if text else "id"
    return text


def _unique_id(base: str, used: set[str]) -> str:
    base = base or "id"
    if base not in used:
        used.add(base)
        return base
    idx = 2
    while f"{base}_{idx}" in used:
        idx += 1
    unique = f"{base}_{idx}"
    used.add(unique)
    return unique


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _init_bounds() -> List[Optional[float]]:
    return [None, None, None, None]  # min_x, max_x, min_y, max_y


def _update_bounds(bounds: List[Optional[float]], x: float, y: float) -> None:
    if bounds[0] is None:
        bounds[0] = x
        bounds[1] = x
        bounds[2] = y
        bounds[3] = y
        return
    bounds[0] = min(bounds[0], x)
    bounds[1] = max(bounds[1], x)
    bounds[2] = min(bounds[2], y)
    bounds[3] = max(bounds[3], y)


def _update_bounds_box(
    bounds: List[Optional[float]],
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_deg: float,
) -> None:
    diag = ((width * width + height * height) ** 0.5) * 0.5
    _update_bounds(bounds, cx - diag, cy - diag)
    _update_bounds(bounds, cx + diag, cy + diag)


def _auto_ranges(bounds: List[Optional[float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if bounds[0] is None:
        return _DEFAULT_X_RANGE, _DEFAULT_Y_RANGE
    min_x, max_x, min_y, max_y = bounds  # type: ignore[misc]
    min_x -= _RANGE_PADDING
    max_x += _RANGE_PADDING
    min_y -= _RANGE_PADDING
    max_y += _RANGE_PADDING

    default_span_x = _DEFAULT_X_RANGE[1] - _DEFAULT_X_RANGE[0]
    default_span_y = _DEFAULT_Y_RANGE[1] - _DEFAULT_Y_RANGE[0]
    min_span_x = default_span_x * _MIN_SPAN_RATIO
    min_span_y = default_span_y * _MIN_SPAN_RATIO

    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x < min_span_x:
        center = (min_x + max_x) / 2
        span_x = min_span_x
        min_x = center - span_x / 2
        max_x = center + span_x / 2
    if span_y < min_span_y:
        center = (min_y + max_y) / 2
        span_y = min_span_y
        min_y = center - span_y / 2
        max_y = center + span_y / 2

    min_x, max_x = _clamp_range(min_x, max_x, _DEFAULT_X_RANGE)
    min_y, max_y = _clamp_range(min_y, max_y, _DEFAULT_Y_RANGE)
    return (min_x, max_x), (min_y, max_y)


def _clamp_range(min_v: float, max_v: float, default_range: Tuple[float, float]) -> Tuple[float, float]:
    default_min, default_max = default_range
    default_span = default_max - default_min
    span = max_v - min_v
    if span >= default_span:
        return default_range
    if min_v < default_min:
        shift = default_min - min_v
        min_v += shift
        max_v += shift
    if max_v > default_max:
        shift = max_v - default_max
        min_v -= shift
        max_v -= shift
    if min_v < default_min or max_v > default_max:
        return default_range
    return min_v, max_v
