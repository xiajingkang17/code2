import subprocess
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    plan_path = root / "examples" / "demo_world3d_plan.json"
    cmd = [
        "python",
        "-m",
        "render.cli",
        "--input",
        str(plan_path),
        "--quality=-ql",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
