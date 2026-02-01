import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

from .config import RenderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a ProblemPlan with Manim")
    parser.add_argument("--input", required=True, help="Path to plan JSON")
    parser.add_argument("--quality", default="-ql", help="Manim quality flag, e.g. -ql")
    parser.add_argument("--renderer", default="cairo", choices=["cairo", "opengl"], help="Manim renderer")
    parser.add_argument("--out", default=None, help="Optional output file or dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = str(Path(args.input).resolve())
    config = RenderConfig(quality=args.quality, renderer=args.renderer, output=args.out)

    env = os.environ.copy()
    env["PLAN_PATH"] = plan_path

    output = config.output or datetime.now().strftime("%Y%m%d_%H%M%S")
    cmd = ["manim", "--renderer", config.renderer, config.quality, config.module, config.scene]
    cmd.extend(["-o", output])
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
