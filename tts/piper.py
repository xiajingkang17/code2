from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from plan.narration import attach_narration, build_narration
from plan.schema import ProblemPlan


@dataclass(frozen=True)
class TTSConfig:
    model_path: str
    bin_path: str = "piper"
    extra_args: str = ""
    overwrite: bool = False


@dataclass(frozen=True)
class AudioEntry:
    q: int
    s: int
    path: str
    duration: float
    text: str
    text_hash: str


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate else 0.0


def _run_piper(text: str, out_path: Path, config: TTSConfig) -> None:
    args = [config.bin_path, "--model", config.model_path, "--output_file", str(out_path)]
    if config.extra_args:
        args.extend(shlex.split(config.extra_args))
    subprocess.run(args, input=text, text=True, check=True)


def synthesize_plan(plan: ProblemPlan, out_dir: Path, config: TTSConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    attach_narration(plan)

    manifest_path = out_dir / "manifest.json"
    existing: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    existing_map: Dict[tuple[int, int], Dict[str, Any]] = {}
    for entry in existing.get("entries", []) if isinstance(existing, dict) else []:
        try:
            existing_map[(int(entry.get("q")), int(entry.get("s")))] = entry
        except Exception:
            continue

    entries: List[AudioEntry] = []
    for qi, q in enumerate(plan.questions, start=1):
        for si, step in enumerate(q.steps, start=1):
            text = step.narration or build_narration(step)
            text_hash = _hash_text(text)
            filename = f"q{qi:02d}_s{si:02d}.wav"
            out_path = out_dir / filename

            cached = existing_map.get((qi, si))
            if (
                out_path.exists()
                and cached
                and cached.get("text_hash") == text_hash
                and not config.overwrite
            ):
                duration = float(cached.get("duration", 0.0)) if cached.get("duration") else _wav_duration(out_path)
            else:
                _run_piper(text, out_path, config)
                duration = _wav_duration(out_path)

            entries.append(
                AudioEntry(
                    q=qi,
                    s=si,
                    path=filename,
                    duration=duration,
                    text=text,
                    text_hash=text_hash,
                )
            )

    payload = {
        "format": "tts_manifest_v1",
        "base_dir": ".",
        "entries": [entry.__dict__ for entry in entries],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def config_from_env() -> TTSConfig:
    model_path = os.environ.get("PIPER_MODEL")
    if not model_path:
        raise RuntimeError("PIPER_MODEL is not set")
    bin_path = os.environ.get("PIPER_BIN", "piper")
    extra_args = os.environ.get("PIPER_ARGS", "")
    overwrite = os.environ.get("PIPER_OVERWRITE", "").lower() in {"1", "true", "yes"}
    return TTSConfig(model_path=model_path, bin_path=bin_path, extra_args=extra_args, overwrite=overwrite)
