from typing import List

from .schema import ProblemPlan


class PlanValidationError(Exception):
    pass


def validate_plan(plan: ProblemPlan) -> List[str]:
    errors: List[str] = []
    if not plan.problem_full_text.strip():
        errors.append("problem_full_text is empty")
    if not plan.stem.strip():
        errors.append("stem is empty")
    if not plan.questions:
        errors.append("questions is empty")

    for qi, q in enumerate(plan.questions, start=1):
        if not q.question_text.strip():
            errors.append(f"Q{qi}: question_text is empty")
        if not q.steps:
            errors.append(f"Q{qi}: steps is empty")
        for si, step in enumerate(q.steps, start=1):
            if not step.line.strip():
                errors.append(f"Q{qi} step {si}: line is empty")
            if not step.subtitle.strip():
                errors.append(f"Q{qi} step {si}: subtitle is empty")

    return errors


def assert_valid(plan: ProblemPlan) -> None:
    errors = validate_plan(plan)
    if errors:
        raise PlanValidationError("; ".join(errors))


def check_alignment(plan: ProblemPlan) -> List[str]:
    mismatches: List[str] = []
    for qi, q in enumerate(plan.questions, start=1):
        for si, step in enumerate(q.steps, start=1):
            if not step.subtitle.strip():
                mismatches.append(f"Q{qi} step {si}: missing subtitle")
    return mismatches
