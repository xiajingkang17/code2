import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from plan.exporter import dump_plan, load_plan
from plan.narration import attach_narration
from plan.schema import ProblemPlan, problem_from_dict
from plan.validator import validate_plan
from plan.llm_solver import ZhipuLLMSolver
from visuals.compiler import compile_plan_visuals
from tts import config_from_env, synthesize_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plan.json from a problem file and render video")
    parser.add_argument("--input", default="problem.txt", help="Path to problem text file")
    parser.add_argument("--solution", default="solution.txt", help="LLM1 output cache file")
    parser.add_argument("--plan", default="plan.json", help="Output plan.json path")
    parser.add_argument("--quality", default="-ql", help="Manim quality flag, e.g. -ql")
    parser.add_argument("--renderer", default="cairo", choices=["cairo", "opengl"], help="Manim renderer")
    parser.add_argument("--out", default=None, help="Optional output file or dir")
    parser.add_argument("--only-llm1", action="store_true", help="Only run LLM1 and write solution.txt")
    parser.add_argument("--only-llm2", action="store_true", help="Only run LLM2 to generate plan.json")
    parser.add_argument("--no-visual", action="store_true", help="Skip visual planning stage")
    parser.add_argument("--only-tts", action="store_true", help="Only run TTS to generate audio manifest")
    parser.add_argument("--only-render", action="store_true", help="Only render video from existing plan.json")
    parser.add_argument("--tts", action="store_true", help="Generate TTS audio and align during render")
    parser.add_argument("--audio-dir", default="media/audio", help="Directory to store TTS audio")
    parser.add_argument("--audio-manifest", default=None, help="Use an existing audio manifest for rendering")
    return parser.parse_args()


def _validate_or_exit(plan: ProblemPlan, *, label: str) -> None:
    errors = validate_plan(plan)
    if not errors:
        return
    details = "\n".join(f"- {err}" for err in errors)
    raise SystemExit(f"{label} validation failed:\n{details}")


def main() -> int:
    # 兼容带 BOM 的 .env（Windows 常见）
    load_dotenv(encoding="utf-8-sig")
    args = parse_args()

    problem_path = Path(args.input)
    if not problem_path.exists():
        raise SystemExit(f"Problem file not found: {problem_path}")

    problem_text = problem_path.read_text(encoding="utf-8").strip()
    if not problem_text:
        raise SystemExit("Problem text is empty")

    solver = ZhipuLLMSolver()
    solution_path = Path(args.solution)
    plan_path = Path(args.plan)

    if args.only_render:
        if not plan_path.exists():
            raise SystemExit(f"plan.json not found: {plan_path}")
        plan = load_plan(plan_path)
        _validate_or_exit(plan, label=f"Plan ({plan_path.resolve()})")
        return _render(plan_path, args.quality, args.renderer, args.out, args.audio_manifest)

    if args.only_llm1:
        solution_text = solver.solve_text(problem_text)
        solution_path.write_text(solution_text, encoding="utf-8")
        print(f"LLM1 solution saved to: {solution_path.resolve()}")
        return 0

    if args.only_llm2:
        if not solution_path.exists():
            raise SystemExit(f"solution.txt not found: {solution_path}")
        solution_text = solution_path.read_text(encoding="utf-8-sig").strip()
        plan_dict = solver.format_json(problem_text, solution_text)
        if _should_generate_visual(args):
            plan_dict = _attach_visuals(solver, plan_dict, solution_text=solution_text)
        plan = problem_from_dict(plan_dict)
        attach_narration(plan)
        dump_plan(plan, plan_path)
        _validate_or_exit(plan, label=f"Plan ({plan_path.resolve()})")
        if args.only_tts:
            manifest = synthesize_plan(plan, Path(args.audio_dir) / plan_path.stem, config_from_env())
            print(f"TTS manifest saved to: {manifest.resolve()}")
        print(f"LLM2 plan saved to: {plan_path.resolve()}")
        return 0

    if args.only_tts:
        if not plan_path.exists():
            raise SystemExit(f"plan.json not found: {plan_path}")
        plan = problem_from_dict(json.loads(plan_path.read_text(encoding="utf-8-sig")))
        attach_narration(plan)
        _validate_or_exit(plan, label=f"Plan ({plan_path.resolve()})")
        manifest = synthesize_plan(plan, Path(args.audio_dir) / plan_path.stem, config_from_env())
        print(f"TTS manifest saved to: {manifest.resolve()}")
        return 0

    if solution_path.exists():
        solution_text = solution_path.read_text(encoding="utf-8-sig").strip()
    else:
        solution_text = solver.solve_text(problem_text)
        solution_path.write_text(solution_text, encoding="utf-8")

    plan_dict = solver.format_json(problem_text, solution_text)
    if _should_generate_visual(args):
        plan_dict = _attach_visuals(solver, plan_dict, solution_text=solution_text)
    plan = problem_from_dict(plan_dict)
    attach_narration(plan)
    dump_plan(plan, plan_path)
    _validate_or_exit(plan, label=f"Plan ({plan_path.resolve()})")

    audio_manifest = None
    if args.tts:
        audio_manifest = synthesize_plan(plan, Path(args.audio_dir) / plan_path.stem, config_from_env())
    return _render(plan_path, args.quality, args.renderer, args.out, str(audio_manifest) if audio_manifest else None)


def _render(plan_path: Path, quality: str, renderer: str, out: str | None, audio_manifest: str | None) -> int:
    env = os.environ.copy()
    env["PLAN_PATH"] = str(plan_path.resolve())
    if audio_manifest:
        env["AUDIO_MANIFEST"] = str(Path(audio_manifest).resolve())

    cmd = ["manim", "--renderer", renderer, quality, "render/scene.py", "ProblemScene"]
    if not out:
        out = datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd.extend(["-o", out])

    return subprocess.call(cmd, env=env)


def _should_generate_visual(args: argparse.Namespace) -> bool:
    if args.no_visual:
        return False
    env = os.environ.get("LLM_VISUAL", "1").strip().lower()
    return env not in {"0", "false", "no", "off"}


def _attach_visuals(solver: ZhipuLLMSolver, plan_dict: dict, *, solution_text: str) -> dict:
    try:
        # 第一步：生成静态visual（左侧图形）
        has_visuals = _plan_has_visuals(plan_dict)
        if not has_visuals:
            visual_dict = solver.format_visuals(plan_dict, solution_text=solution_text)
            plan_dict = solver.merge_visuals(plan_dict, visual_dict)
        plan_dict = compile_plan_visuals(plan_dict)
        
        # 第二步：生成动态visual_sequence（步骤级变换）
        try:
            visual_seq_dict = solver.format_visual_sequence(plan_dict, solution_text=solution_text)
            plan_dict = solver.merge_visual_sequence(plan_dict, visual_seq_dict)
            print("Visual sequence planning completed")
        except Exception as exc:
            print(f"Visual sequence planning skipped: {exc}")
        
        return plan_dict
    except Exception as exc:
        print(f"Visual planning skipped: {exc}")
        return plan_dict


def _plan_has_visuals(plan_dict: dict) -> bool:
    questions = plan_dict.get("questions", [])
    if not isinstance(questions, list) or not questions:
        return False
    return all("visual" in q for q in questions if isinstance(q, dict))


if __name__ == "__main__":
    raise SystemExit(main())
