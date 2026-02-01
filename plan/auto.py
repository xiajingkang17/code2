import argparse
from pathlib import Path

from dotenv import load_dotenv

from .exporter import dump_plan
from .llm_solver import ZhipuLLMSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ProblemPlan JSON from a problem text")
    parser.add_argument("--input", help="Path to a UTF-8 text file")
    parser.add_argument("--text", help="Problem text directly")
    parser.add_argument("--output", required=True, help="Output plan.json path")
    return parser.parse_args()


def main() -> int:
    # 自动加载 .env 中的配置（如 ZHIPU_API_KEY），兼容带 BOM
    load_dotenv(encoding="utf-8-sig")

    args = parse_args()
    if not args.input and not args.text:
        raise SystemExit("Either --input or --text is required")

    if args.text:
        problem_text = args.text.strip()
    else:
        problem_text = Path(args.input).read_text(encoding="utf-8")

    solver = ZhipuLLMSolver()
    plan = solver.solve(problem_text)
    dump_plan(plan, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
