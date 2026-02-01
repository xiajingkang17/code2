from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from manim import Mobject

from layout import Theme
from plan.schema import StepVisual
from visuals.library import build_from_spec
from visuals.compiler import compile_visual_spec


@dataclass(frozen=True)
class VisualResult:
    mobject: Mobject
    warning: Optional[str] = None


def build_visual(
    spec: Optional[Dict[str, Any]],
    width: float,
    height: float,
    *,
    theme: Theme,
) -> VisualResult:
    compiled = compile_visual_spec(spec)
    result = build_from_spec(compiled, width, height, theme=theme)
    return VisualResult(result.mobject, warning=result.warning)


def build_visual_with_dict(
    spec: Optional[Dict[str, Any]],
    width: float,
    height: float,
    *,
    theme: Theme,
) -> tuple[VisualResult, Dict[str, Mobject]]:
    compiled = compile_visual_spec(spec)
    result = build_from_spec(compiled, width, height, theme=theme)
    return VisualResult(result.mobject, warning=result.warning), result.id_map


def apply_visual_transform(
    mobject_dict: Dict[str, Mobject],
    transform: StepVisual,
) -> None:
    return
