import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when Manim loads this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plan.exporter import load_plan
from plan.validator import validate_plan
from template.flow import ProblemSceneBase


class ProblemScene(ProblemSceneBase):
    def construct(self) -> None:
        plan_path = os.environ.get("PLAN_PATH")
        if not plan_path:
            raise RuntimeError("PLAN_PATH is not set")
        plan = load_plan(plan_path)
        errors = validate_plan(plan)
        if errors:
            details = "\n".join(f"- {err}" for err in errors)
            raise RuntimeError(f"Plan validation failed:\n{details}")
        self.play_problem(plan)
