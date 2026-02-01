from __future__ import annotations

import re
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .schema import ProblemPlan, Step


_STEP_PREFIX_RE = re.compile(r"^\s*第?\s*\d+\s*步[:：]\s*")
_INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")
_LATEX_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")
_LATEX_NROOT_RE = re.compile(r"\\sqrt\[(.+?)\]\{([^{}]+)\}")
_LATEX_SUM_RE = re.compile(r"\\sum_\{([^{}]+)\}\^\{([^{}]+)\}")
_LATEX_SUM_RE_ALT = re.compile(r"\\sum\^\{([^{}]+)\}_\{([^{}]+)\}")
_LATEX_PROD_RE = re.compile(r"\\prod_\{([^{}]+)\}\^\{([^{}]+)\}")
_LATEX_PROD_RE_ALT = re.compile(r"\\prod\^\{([^{}]+)\}_\{([^{}]+)\}")
_LATEX_VEC_RE = re.compile(r"\\vec\{([^{}]+)\}")
_LATEX_ARROW_RE = re.compile(r"\\overrightarrow\{([^{}]+)\}")
_LATEX_DERIV_RE = re.compile(r"\\frac\{d\}\{d([a-zA-Z])\}")
_LATEX_PDERIV_RE = re.compile(r"\\frac\{\\partial\}\{\\partial\s*([a-zA-Z])\}")
_SUP_BRACE_RE = re.compile(r"\^\{([^{}]+)\}")
_SUP_RE = re.compile(r"\^([A-Za-z0-9.+-]+)")
_SUB_BRACE_RE = re.compile(r"_\{([^{}]+)\}")
_SUB_RE = re.compile(r"_([A-Za-z0-9.+-]+)")
_NEG_NUMBER_RE = re.compile(r"(?:(?<=^)|(?<=[=,(+\-*/\s]))-\s*([0-9]+(?:\.[0-9]+)?)")
_PRIME_RE = re.compile(r"([A-Za-z])('+)")


@dataclass(frozen=True)
class _DictRule:
    pattern: re.Pattern
    replace: str


@dataclass(frozen=True)
class _TtsDict:
    raw_literal: dict[str, str]
    raw_regex: list[_DictRule]
    speech_literal: dict[str, str]
    speech_regex: list[_DictRule]

_LATEX_COMMANDS = {
    "cdot": "乘",
    "times": "乘",
    "div": "除以",
    "pm": "正负",
    "mp": "负正",
    "le": "小于等于",
    "ge": "大于等于",
    "neq": "不等于",
    "approx": "约等于",
    "sim": "相似于",
    "propto": "成正比于",
    "infty": "无穷大",
    "sum": "求和",
    "prod": "连乘",
    "int": "积分",
    "lim": "极限",
    "sin": "正弦",
    "cos": "余弦",
    "tan": "正切",
    "cot": "余切",
    "ln": "自然对数",
    "log": "对数",
    "min": "最小值",
    "max": "最大值",
    "cdots": "省略号",
    "ldots": "省略号",
}

_LATEX_GREEK = {
    "alpha": "阿尔法",
    "beta": "贝塔",
    "gamma": "伽马",
    "delta": "德尔塔",
    "epsilon": "艾普西龙",
    "zeta": "泽塔",
    "eta": "埃塔",
    "theta": "西塔",
    "iota": "艾欧塔",
    "kappa": "卡帕",
    "lambda": "拉姆达",
    "mu": "缪",
    "nu": "纽",
    "xi": "克西",
    "pi": "派",
    "rho": "柔",
    "sigma": "西格玛",
    "tau": "陶",
    "upsilon": "宇普西龙",
    "phi": "斐",
    "chi": "凯",
    "psi": "普西",
    "omega": "欧米伽",
}


def _strip_step_prefix(text: str) -> str:
    return _STEP_PREFIX_RE.sub("", text).strip()


def _normalize_math_for_speech(text: str) -> str:
    if not text:
        return text
    text = text.replace("$$", "$")
    if "$" not in text and "\\" not in text and "^" not in text and "_" not in text:
        return _normalize_spacing(_apply_dicts_speech(text))

    if "$" in text:
        parts: list[str] = []
        last = 0
        for match in _INLINE_MATH_RE.finditer(text):
            parts.append(text[last:match.start()])
            parts.append(_latex_fragment_to_speech(match.group(1)))
            last = match.end()
        parts.append(text[last:])
        combined = "".join(parts)
    else:
        combined = _latex_fragment_to_speech(text)

    combined = combined.replace("$", "")
    combined = _apply_dicts_speech(combined)
    return _normalize_spacing(combined)


def _normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _apply_literal(text: str, literal: dict[str, str]) -> str:
    if not literal:
        return text
    for key in sorted(literal.keys(), key=len, reverse=True):
        text = text.replace(key, literal[key])
    return text


def _apply_rules(text: str, rules: list[_DictRule]) -> str:
    if not rules:
        return text
    for rule in rules:
        text = rule.pattern.sub(rule.replace, text)
    return text


def _apply_dicts_raw(text: str) -> str:
    for d in _load_tts_dicts():
        text = _apply_literal(text, d.raw_literal)
        text = _apply_rules(text, d.raw_regex)
    return text


def _apply_dicts_speech(text: str) -> str:
    for d in _load_tts_dicts():
        text = _apply_literal(text, d.speech_literal)
        text = _apply_rules(text, d.speech_regex)
    return text


def _compile_rules(items: list[dict]) -> list[_DictRule]:
    rules: list[_DictRule] = []
    for item in items or []:
        pattern = item.get("pattern")
        if not pattern:
            continue
        replace = str(item.get("replace", ""))
        flags_raw = str(item.get("flags", ""))
        flags = 0
        if "i" in flags_raw.lower():
            flags |= re.IGNORECASE
        if "m" in flags_raw.lower():
            flags |= re.MULTILINE
        if "s" in flags_raw.lower():
            flags |= re.DOTALL
        try:
            compiled = re.compile(pattern, flags)
        except re.error:
            continue
        rules.append(_DictRule(pattern=compiled, replace=replace))
    return rules


@lru_cache(maxsize=1)
def _load_tts_dicts() -> list[_TtsDict]:
    names = os.environ.get("TTS_DICTS", "math").strip()
    extra = os.environ.get("TTS_DICT_PATHS", "").strip()
    candidates: list[Path] = []
    if names:
        base = Path(__file__).resolve().parents[1] / "tts"
        for name in re.split(r"[;,]", names):
            key = name.strip().lower()
            if not key:
                continue
            if key in {"math", "physics"}:
                candidates.append(base / f"tts_dict_{key}.json")
            else:
                candidates.append(Path(key))
    if extra:
        for raw in re.split(r"[;,]", extra):
            raw = raw.strip()
            if raw:
                candidates.append(Path(raw))

    dicts: list[_TtsDict] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw = payload.get("raw", {}) if isinstance(payload, dict) else {}
        speech = payload.get("speech", {}) if isinstance(payload, dict) else {}
        dicts.append(
            _TtsDict(
                raw_literal=dict(raw.get("literal", {}) or {}),
                raw_regex=_compile_rules(list(raw.get("regex", []) or [])),
                speech_literal=dict(speech.get("literal", {}) or {}),
                speech_regex=_compile_rules(list(speech.get("regex", []) or [])),
            )
        )
    return dicts


def _replace_math_operators(text: str) -> str:
    if not text:
        return text
    text = _NEG_NUMBER_RE.sub(r"负\1", text)
    text = text.replace("≥", " 大于等于 ").replace("≤", " 小于等于 ")
    text = text.replace("≠", " 不等于 ").replace("≈", " 约等于 ")
    text = text.replace("=", " 等于 ")
    text = text.replace("+", " 加 ")
    text = text.replace("-", " 减 ")
    text = text.replace("*", " 乘 ")
    text = text.replace("/", " 除以 ")
    return text


def _render_sum_prod(lower: str, upper: str, action: str) -> str:
    lower = lower.strip()
    upper = upper.strip()
    if "=" in lower:
        var, start = lower.split("=", 1)
        var = var.strip()
        start = start.strip()
        if var and start and upper:
            return f"对 {var} 从 {start} 到 {upper} {action}"
    if lower and upper:
        return f"对 {lower} 到 {upper} {action}"
    if lower:
        return f"对 {lower} {action}"
    return action


def _latex_fragment_to_speech(fragment: str) -> str:
    if not fragment:
        return fragment
    text = _apply_dicts_raw(fragment)
    text = text.replace("\\,", " ").replace("\\;", " ").replace("\\:", " ")
    text = text.replace("\\quad", " ").replace("\\qquad", " ")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)

    text = _LATEX_DERIV_RE.sub(r"对 \1 求导", text)
    text = _LATEX_PDERIV_RE.sub(r"对 \1 偏导", text)
    text = _LATEX_SUM_RE.sub(lambda m: _render_sum_prod(m.group(1), m.group(2), "求和"), text)
    text = _LATEX_SUM_RE_ALT.sub(lambda m: _render_sum_prod(m.group(2), m.group(1), "求和"), text)
    text = _LATEX_PROD_RE.sub(lambda m: _render_sum_prod(m.group(1), m.group(2), "连乘"), text)
    text = _LATEX_PROD_RE_ALT.sub(lambda m: _render_sum_prod(m.group(2), m.group(1), "连乘"), text)
    text = _LATEX_VEC_RE.sub(r"向量 \1", text)
    text = _LATEX_ARROW_RE.sub(r"向量 \1", text)

    prev = None
    while prev != text:
        prev = text
        text = _LATEX_FRAC_RE.sub(r"\1 分之 \2", text)
        text = _LATEX_NROOT_RE.sub(r"\1 次根号 \2", text)
        text = _LATEX_SQRT_RE.sub(r"根号 \1", text)

    for cmd, speech in _LATEX_COMMANDS.items():
        text = re.sub(rf"\\{cmd}\b", speech, text)
    for cmd, speech in _LATEX_GREEK.items():
        text = re.sub(rf"\\{cmd}\b", speech, text)

    text = _PRIME_RE.sub(lambda m: f"{m.group(1)} 的 {len(m.group(2))} 阶导", text)
    text = _SUP_BRACE_RE.sub(r" 的 \1 次方", text)
    text = _SUP_RE.sub(r" 的 \1 次方", text)
    text = _SUB_BRACE_RE.sub(r" 下标 \1", text)
    text = _SUB_RE.sub(r" 下标 \1", text)

    text = _replace_math_operators(text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    return text


def build_narration(step: Step) -> str:
    base = step.subtitle.strip() if step.subtitle else step.line.strip()
    base = _strip_step_prefix(base)
    return _normalize_math_for_speech(base)


def attach_narration(plan: ProblemPlan) -> ProblemPlan:
    for q in plan.questions:
        for step in q.steps:
            if step.narration:
                continue
            step.narration = build_narration(step)
    return plan
