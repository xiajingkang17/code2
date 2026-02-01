from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .theme import Constraints


def _looks_like_math(text: str) -> bool:
    return "\\" in text or "^" in text or "_" in text or "$" in text


@dataclass(frozen=True)
class LayoutMetrics:
    total_chars: int
    line_count: int
    formula_ratio: float
    has_multiline_math: bool


@dataclass(frozen=True)
class LayoutDecision:
    strategy: str
    font_scale: float
    max_lines: int


def compute_metrics(texts: Iterable[str], line_count: int) -> LayoutMetrics:
    total_chars = 0
    formula_chars = 0
    has_multiline_math = False
    for t in texts:
        if not t:
            continue
        total_chars += len(t)
        if _looks_like_math(t):
            formula_chars += len(t)
        if "\\\\" in t or "\n" in t or "\\begin" in t:
            has_multiline_math = True
    ratio = (formula_chars / total_chars) if total_chars > 0 else 0.0
    return LayoutMetrics(
        total_chars=total_chars,
        line_count=line_count,
        formula_ratio=ratio,
        has_multiline_math=has_multiline_math,
    )


def decide_layout(metrics: LayoutMetrics, constraints: Constraints) -> LayoutDecision:
    dense = metrics.line_count > constraints.max_lines + 1 or metrics.total_chars > 260
    math_heavy = metrics.formula_ratio > 0.45 or metrics.has_multiline_math

    if dense or math_heavy:
        return LayoutDecision(strategy="compact", font_scale=0.9, max_lines=constraints.max_lines)
    if metrics.total_chars < 140 and metrics.line_count <= max(2, constraints.max_lines - 1):
        return LayoutDecision(strategy="relaxed", font_scale=1.05, max_lines=constraints.max_lines)
    return LayoutDecision(strategy="standard", font_scale=1.0, max_lines=constraints.max_lines)
