from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    if not isinstance(value, (int, float)):
        return fallback
    if value != value:
        return fallback
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class Theme:
    font: str = "Microsoft YaHei"
    font_size_scale: float = 1.0
    line_spacing: float = 1.0
    panel_padding: float = 0.2
    subtitle_bg_opacity: float = 0.0
    accent_color: str = "#ffffff"


@dataclass(frozen=True)
class Constraints:
    max_width_ratio: float = 0.9
    max_lines: int = 6
    min_font_size: float = 20.0
    min_margin: float = 0.35
    safe_top: float = 0.6
    safe_bottom: float = 0.6


def _coerce_theme(base: Theme, override: Dict[str, Any]) -> Theme:
    font = str(override.get("font", base.font)) if override.get("font") is not None else base.font
    font_size_scale = _clamp(override.get("font_size_scale", base.font_size_scale), 0.7, 1.4, base.font_size_scale)
    line_spacing = _clamp(override.get("line_spacing", base.line_spacing), 0.7, 2.0, base.line_spacing)
    panel_padding = _clamp(override.get("panel_padding", base.panel_padding), 0.05, 1.0, base.panel_padding)
    subtitle_bg_opacity = _clamp(
        override.get("subtitle_bg_opacity", base.subtitle_bg_opacity), 0.0, 0.9, base.subtitle_bg_opacity
    )
    accent_color = str(override.get("accent_color", base.accent_color)) if override.get("accent_color") is not None else base.accent_color
    return replace(
        base,
        font=font,
        font_size_scale=font_size_scale,
        line_spacing=line_spacing,
        panel_padding=panel_padding,
        subtitle_bg_opacity=subtitle_bg_opacity,
        accent_color=accent_color,
    )


def _coerce_constraints(base: Constraints, override: Dict[str, Any]) -> Constraints:
    max_width_ratio = _clamp(override.get("max_width_ratio", base.max_width_ratio), 0.6, 0.98, base.max_width_ratio)
    max_lines_raw = override.get("max_lines", base.max_lines)
    if isinstance(max_lines_raw, (int, float)):
        max_lines = int(max(1, min(12, max_lines_raw)))
    else:
        max_lines = base.max_lines
    min_font_size = _clamp(override.get("min_font_size", base.min_font_size), 12.0, 48.0, base.min_font_size)
    min_margin = _clamp(override.get("min_margin", base.min_margin), 0.1, 1.5, base.min_margin)
    safe_top = _clamp(override.get("safe_top", base.safe_top), 0.2, 2.0, base.safe_top)
    safe_bottom = _clamp(override.get("safe_bottom", base.safe_bottom), 0.2, 2.0, base.safe_bottom)
    return replace(
        base,
        max_width_ratio=max_width_ratio,
        max_lines=max_lines,
        min_font_size=min_font_size,
        min_margin=min_margin,
        safe_top=safe_top,
        safe_bottom=safe_bottom,
    )


def apply_overrides(
    base_theme: Theme,
    base_constraints: Constraints,
    overrides: Optional[Dict[str, Any]],
) -> Tuple[Theme, Constraints]:
    if not overrides:
        return base_theme, base_constraints
    theme_overrides = dict(overrides.get("theme", {}))
    constraints_overrides = dict(overrides.get("constraints", {}))
    # Allow flat overrides as well.
    for key, value in overrides.items():
        if key in Theme.__dataclass_fields__:
            theme_overrides.setdefault(key, value)
        if key in Constraints.__dataclass_fields__:
            constraints_overrides.setdefault(key, value)
    return _coerce_theme(base_theme, theme_overrides), _coerce_constraints(base_constraints, constraints_overrides)
