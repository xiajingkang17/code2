from __future__ import annotations

from typing import Protocol

from .schema import ProblemPlan


class PlanSolver(Protocol):
    def solve(self, problem_text: str) -> ProblemPlan:  # pragma: no cover - interface only
        ...


class NotImplementedSolver:
    def solve(self, problem_text: str) -> ProblemPlan:
        raise NotImplementedError("Connect your LLM or rule-based solver here")
