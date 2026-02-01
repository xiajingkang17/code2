import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schema import ProblemPlan, problem_from_dict
from .solver import PlanSolver


@dataclass
class ZhipuConfig:
    api_key: str
    model: str = "glm-4.7"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    timeout_s: int = 60


class ZhipuLLMSolver(PlanSolver):
    def __init__(self, config: Optional[ZhipuConfig] = None) -> None:
        if config is None:
            api_key = os.environ.get("ZHIPU_API_KEY") or os.environ.get("ZAI_API_KEY")
            if not api_key:
                raise RuntimeError("ZHIPU_API_KEY is not set")
            base_url = os.environ.get("ZHIPU_BASE_URL") or os.environ.get("ZAI_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/"
            model = os.environ.get("ZHIPU_MODEL") or "glm-4.7"
            timeout_s = int(os.environ.get("ZHIPU_TIMEOUT", "60"))
            config = ZhipuConfig(api_key=api_key, base_url=base_url, model=model, timeout_s=timeout_s)
        self.config = config
        self.retries = int(os.environ.get("ZHIPU_RETRIES", "2"))
        self.backoff_s = float(os.environ.get("ZHIPU_BACKOFF", "2"))

    def solve(self, problem_text: str) -> ProblemPlan:
        solution_text = self.solve_text(problem_text)
        plan_dict = self.format_json(problem_text, solution_text)
        return problem_from_dict(plan_dict)

    def solve_text(self, problem_text: str) -> str:
        payload = _make_payload(self.config.model, _SOLVE_SYSTEM_PROMPT, problem_text)
        data = _post_json(
            self.config.base_url + "chat/completions",
            payload,
            self.config.api_key,
            self.config.timeout_s,
            retries=self.retries,
            backoff_s=self.backoff_s,
        )
        return _extract_content(data)

    def format_json(self, problem_text: str, solution_text: str) -> Dict[str, Any]:
        user_content = f"题目：\n{problem_text}\n\n解题文本：\n{solution_text}"
        payload = _make_payload(self.config.model, _FORMAT_SYSTEM_PROMPT, user_content)
        data = _post_json(
            self.config.base_url + "chat/completions",
            payload,
            self.config.api_key,
            self.config.timeout_s,
            retries=self.retries,
            backoff_s=self.backoff_s,
        )
        content = _extract_content(data)
        plan_dict = _extract_json(content)
        # Disabled post-processing to inspect raw LLM2 output.
        return plan_dict

    def format_visuals(self, plan_dict: Dict[str, Any], solution_text: Optional[str] = None) -> Dict[str, Any]:
        visual_input = _trim_plan_for_visual(plan_dict, solution_text=solution_text)
        user_content = json.dumps(visual_input, ensure_ascii=False)
        payload = _make_payload(self.config.model, _VISUAL_SYSTEM_PROMPT, user_content)
        data = _post_json(
            self.config.base_url + "chat/completions",
            payload,
            self.config.api_key,
            self.config.timeout_s,
            retries=self.retries,
            backoff_s=self.backoff_s,
        )
        content = _extract_content(data)
        return _extract_json(content)

    def merge_visuals(self, plan_dict: Dict[str, Any], visual_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not visual_dict:
            return plan_dict

        questions = plan_dict.get("questions", [])
        if not isinstance(questions, list) or not questions:
            return plan_dict

        if isinstance(visual_dict.get("questions"), list):
            for idx, visual_item in enumerate(visual_dict["questions"]):
                if idx >= len(questions):
                    break
                if isinstance(visual_item, dict) and "visual" in visual_item:
                    visual_item = visual_item.get("visual")
                if isinstance(visual_item, dict) and visual_item:
                    questions[idx]["visual"] = visual_item
            return plan_dict

        visuals = None
        if isinstance(visual_dict.get("visuals"), list):
            visuals = visual_dict.get("visuals")
        elif isinstance(visual_dict.get("visual"), dict):
            visuals = [visual_dict.get("visual")]

        if isinstance(visuals, list):
            for idx, visual_item in enumerate(visuals):
                if idx >= len(questions):
                    break
                if isinstance(visual_item, dict) and visual_item:
                    questions[idx]["visual"] = visual_item
        return plan_dict

    def format_visual_sequence(
        self, plan_dict: Dict[str, Any], solution_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成步骤级的visual_transform序列
        对于每个问题的每个步骤，生成相应的图形变换描述
        
        :param plan_dict: 包含questions/steps的规划字典
        :param solution_text: 可选的原始解题文本
        :return: 包含visual_sequence的字典，格式为：
                 {"questions": [{"question_index": 0, "visual_transforms": [...]}, ...]}
        """
        visual_seq_input = _prepare_visual_sequence_input(plan_dict, solution_text=solution_text)
        user_content = json.dumps(visual_seq_input, ensure_ascii=False)
        payload = _make_payload(self.config.model, _VISUAL_SEQUENCE_SYSTEM_PROMPT, user_content)
        data = _post_json(
            self.config.base_url + "chat/completions",
            payload,
            self.config.api_key,
            self.config.timeout_s,
            retries=self.retries,
            backoff_s=self.backoff_s,
        )
        content = _extract_content(data)
        return _extract_json(content)

    def merge_visual_sequence(
        self, plan_dict: Dict[str, Any], visual_seq_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将visual_sequence合并到plan中
        为每个步骤添加visual_transform字段
        
        :param plan_dict: 原始规划字典
        :param visual_seq_dict: 包含visual_sequence的字典
        :return: 合并后的规划字典
        """
        if not visual_seq_dict:
            return plan_dict

        questions = plan_dict.get("questions", [])
        if not isinstance(questions, list):
            return plan_dict

        visual_questions = visual_seq_dict.get("questions", [])
        if not isinstance(visual_questions, list):
            return plan_dict

        # 为每个问题的步骤添加visual_transform
        for q_idx, visual_q in enumerate(visual_questions):
            if q_idx >= len(questions):
                break
            if not isinstance(visual_q, dict):
                continue

            visual_transforms_list = visual_q.get("visual_transforms", [])
            if not isinstance(visual_transforms_list, list):
                continue

            steps = questions[q_idx].get("steps", [])
            if not isinstance(steps, list):
                continue

            # 为每个步骤分配相应的visual_transform
            for step_idx, visual_transforms in enumerate(visual_transforms_list):
                if step_idx >= len(steps):
                    break
                # 如果已有 visual_transform，则优先保留
                if isinstance(steps[step_idx], dict) and steps[step_idx].get("visual_transform"):
                    continue
                
                # visual_transforms 可能是 null、单个dict、或dict列表
                if visual_transforms is None:
                    continue
                
                # 如果是单个dict（单个变换），转换为列表格式
                if isinstance(visual_transforms, dict):
                    steps[step_idx]["visual_transform"] = [visual_transforms]
                # 如果已经是列表，直接使用
                elif isinstance(visual_transforms, list) and visual_transforms:
                    steps[step_idx]["visual_transform"] = visual_transforms

        return _sanitize_visual_transforms(plan_dict)


def _make_payload(model: str, system_prompt: str, user_content: str) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }


def _post_json(
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    timeout_s: int,
    retries: int = 2,
    backoff_s: float = 2.0,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(backoff_s * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("Request failed")


def _extract_content(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - best effort
        raise RuntimeError(f"Unexpected response format: {data}") from exc


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _trim_plan_for_visual(plan: Dict[str, Any], *, solution_text: Optional[str] = None) -> Dict[str, Any]:
    questions = []
    for q in plan.get("questions", []):
        if not isinstance(q, dict):
            continue
        steps = q.get("steps", [])
        step_lines = []
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    line = str(step.get("line", "")).strip()
                    if line:
                        step_lines.append(line)
        questions.append(
            {
                "question_text": str(q.get("question_text", "")),
                "analysis": q.get("analysis", {}),
                "steps": step_lines,
                "model_spec": q.get("model_spec"),
                "diagram_spec": q.get("diagram_spec"),
                "motion_spec": q.get("motion_spec"),
            }
        )
    visual_hints = _extract_visual_hints(solution_text) if solution_text else []
    out = {
        "stem": str(plan.get("stem", "")),
        "problem_full_text": str(plan.get("problem_full_text", "")),
        "questions": questions,
    }
    if visual_hints:
        out["visual_hints"] = visual_hints
    return out


def _extract_visual_hints(text: Optional[str]) -> List[str]:
    if not text:
        return []
    hints: List[str] = []
    sections = ["图形要素", "视觉意图", "ModelSpec", "DiagramSpec", "MotionSpec", "绘图规划"]
    for section in sections:
        pattern = rf"【{section}】"
        for match in re.finditer(pattern, text):
            start = match.end()
            end = len(text)
            next_match = re.search(r"【", text[start:])
            if next_match:
                end = start + next_match.start()
            chunk = text[start:end].strip()
            if chunk:
                hints.append(f"{section}:\n{chunk}")
    return hints


def _postprocess_plan_dict(plan: Dict[str, Any]) -> Dict[str, Any]:
    # 为含 LaTeX 的片段自动补齐 $...$，提升混排渲染成功率
    def fix_text(value: str) -> str:
        return _wrap_latex_outside_dollars(value)

    stem = fix_text(str(plan.get("stem", "")))
    stem = _clean_stem(stem)
    plan["stem"] = stem
    for q in plan.get("questions", []):
        q["question_text"] = _clean_question_text(fix_text(str(q.get("question_text", ""))))
        analysis = q.get("analysis", {}) or {}
        analysis["formulas"] = [fix_text(str(x)) for x in analysis.get("formulas", [])]
        analysis["conditions"] = [fix_text(str(x)) for x in analysis.get("conditions", [])]
        analysis["strategy"] = [fix_text(str(x)) for x in analysis.get("strategy", [])]
        q["analysis"] = analysis
        raw_steps = []
        for step in q.get("steps", []):
            line = str(step.get("line", ""))
            subtitle = str(step.get("subtitle", ""))
            raw_steps.append({"line": line, "subtitle": subtitle})
        merged_steps = _merge_step_fragments(raw_steps)
        fixed_steps = []
        for step in merged_steps:
            line = fix_text(str(step.get("line", "")))
            subtitle = fix_text(str(step.get("subtitle", "")))
            fixed_steps.append({"line": line, "subtitle": subtitle})
        q["steps"] = _merge_step_fragments(_split_steps(fixed_steps))
    # 统一生成带换行的 problem_full_text（题干 + 每个小问一行）
    questions = [q.get("question_text", "") for q in plan.get("questions", [])]
    if stem and questions:
        normalized = []
        for idx, qt in enumerate(questions, start=1):
            text = str(qt).strip()
            if not text:
                continue
            # 若缺少 (1)(2) 等编号则补齐
            if not re.match(r"^\(?\d+\)?", text):
                text = f"({idx}) {text}"
            normalized.append(text)
        plan["problem_full_text"] = stem + "\n" + "\n".join(normalized)
    else:
        plan["problem_full_text"] = fix_text(str(plan.get("problem_full_text", "")))
    return plan


def _split_steps(steps: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    # 将过长或含多段的 step 拆成多行，便于逐行书写
    if not steps:
        return steps

    new_steps: list[Dict[str, Any]] = []
    for step in steps:
        line = str(step.get("line", "")).strip()
        subtitle = str(step.get("subtitle", "")).strip()

        line_parts = _split_text_parts(line)
        subtitle_parts = _split_text_parts(subtitle)

        if len(line_parts) <= 1 and len(subtitle_parts) <= 1:
            new_steps.append(step)
            continue

        # 对齐数量
        count = max(len(line_parts), len(subtitle_parts))
        if not line_parts:
            line_parts = [line] * count
        if not subtitle_parts:
            subtitle_parts = [subtitle] * count
        if len(line_parts) < count:
            line_parts.extend([line_parts[-1]] * (count - len(line_parts)))
        if len(subtitle_parts) < count:
            subtitle_parts.extend([subtitle_parts[-1]] * (count - len(subtitle_parts)))

        for i in range(count):
            new_steps.append({"line": line_parts[i], "subtitle": subtitle_parts[i]})

    return new_steps


def _split_text_parts(text: str) -> list[str]:
    # 先按 $...$ / $$...$$ 分块，再按中文标点切分
    if not text:
        return []
    parts: list[str] = []
    chunks = re.split(r"(\\$\\$[\\s\\S]*?\\$\\$|\\$[\\s\\S]*?\\$)", text)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if (chunk.startswith("$$") and chunk.endswith("$$")) or (chunk.startswith("$") and chunk.endswith("$")):
            parts.append(chunk)
            continue
        # 细分：中文分号/句号/换行
        for seg in re.split(r"[；。\\n]+", chunk):
            seg = seg.strip()
            if seg:
                parts.append(seg)
    return parts


_LATEX_CMD_RE = re.compile(r"(\\[a-zA-Z]+(?:\{[^}]*\}){0,3})")
_BARE_LATEX_CMD_RE = re.compile(
    r"(?<!\\)(frac|sqrt|sum|prod|int|lim|ln|log|sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|min|max|arg|deg|"
    r"alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|"
    r"psi|omega|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega|hbar|partial|nabla|vec|overrightarrow|hat|bar|tilde|"
    r"dot|ddot|left|right|mathrm|mathbf)"
    r"(?=\\s*\\{)"
)
_BARE_LATEX_OP_RE = re.compile(
    r"(?<!\\)(cdot|times|div|pm|mp|le|ge|neq|approx|sim|propto|in|subset|subseteq|cup|cap|setminus|forall|exists|therefore|"
    r"because|infty|rightarrow|leftarrow|leftrightarrow|rightleftharpoons)\\b"
)
_MATH_HINT_RE = re.compile(r"(\\[a-zA-Z]+|\\^|_|=|\\(|\\)|\\[|\\]|\\{\\}|\\d+\\s*[\\+\\-\\*/])")
_PROMPT_PREFIX_RE = re.compile(r"(请(?:你)?(?:解答|回答)下列问题[:：]?)+")
_SCORE_RE = re.compile(r"[（(]\\s*\\d+\\s*分\\s*[）)]")
_LEADING_INDEX_RE = re.compile(r"^\(?\d+\)?[.、．)]?\s*")
_TRAILING_PUNCT_RE = re.compile(r"[，,;；:：。\\.]+$")


def _is_incomplete_line(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if stripped.count("$") % 2 == 1:
        return True
    if stripped.count("{") > stripped.count("}"):
        return True
    if stripped.endswith(("+", "-", "=", "(", "[", "{", "\\")):
        return True
    return False


def _looks_like_continuation(left: str, right: str, left_sub: str, right_sub: str) -> bool:
    if not right:
        return False
    if left_sub and right_sub and left_sub != right_sub:
        return False
    left_strip = left.strip()
    right_strip = right.lstrip()
    if left_strip.endswith(("=", "+", "-", "(", "[", "{", "$")):
        return True
    if right_strip.startswith(("frac", "\\frac", "sqrt", "\\sqrt", "\\", "^", "_", ")", "]", "}")):
        return True
    if re.match(r"^[=+\\-*/]", right_strip):
        return True
    if _MATH_HINT_RE.search(right_strip) and len(right_strip) < 24 and right_strip.count("$") == 0:
        return True
    return False


def _join_fragments(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    right_strip = right.lstrip()
    left_strip = left.rstrip()
    if right_strip.startswith(("cdot", "times", ")", "}", "]", "^", "_", "$", "\\")):
        return f"{left_strip}{right_strip}"
    if left_strip.endswith(("{", "(", "[", "$", "\\")):
        return f"{left_strip}{right_strip}"
    return f"{left_strip} {right_strip}"


def _merge_step_fragments(steps: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    if not steps:
        return steps
    merged: list[Dict[str, Any]] = []
    i = 0
    while i < len(steps):
        line = str(steps[i].get("line", "")).strip()
        subtitle = str(steps[i].get("subtitle", "")).strip()
        j = i
        while j + 1 < len(steps):
            next_line = str(steps[j + 1].get("line", "")).strip()
            next_sub = str(steps[j + 1].get("subtitle", "")).strip()
            if not (_is_incomplete_line(line) or _looks_like_continuation(line, next_line, subtitle, next_sub)):
                break
            j += 1
            line = _join_fragments(line, next_line)
            subtitle = _join_fragments(subtitle, next_sub)
        merged.append({"line": line, "subtitle": subtitle})
        i = j + 1
    return merged


def _normalize_dollars(text: str) -> str:
    # 归一化多个 $，并修补奇数个 $ 的情况
    if not text:
        return text
    text = re.sub(r"\${3,}", "$$", text)
    if text.count("$") % 2 == 1:
        # 尝试包整段，避免裸露公式
        text = f"${text.strip('$')}$"
    return text


def _wrap_latex_outside_dollars(text: str) -> str:
    # 仅处理未被 $...$ 包裹的段落
    if not text:
        return text
    text = _normalize_dollars(text)
    parts = text.split("$")
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        # 修补缺失反斜杠的常见命令
        chunk = _BARE_LATEX_CMD_RE.sub(r"\\\1", chunk)
        chunk = _BARE_LATEX_OP_RE.sub(r"\\\1", chunk)
        # 包裹 LaTeX 命令片段
        chunk = _LATEX_CMD_RE.sub(r"$\1$", chunk)
        # 对明显是数学表达式但没被包裹的片段整体加 $
        if _MATH_HINT_RE.search(chunk):
            chunk = f"${chunk}$"
        parts[i] = chunk
    return "$".join(parts)


def _clean_stem(text: str) -> str:
    if not text:
        return text
    text = _PROMPT_PREFIX_RE.sub("", text)
    text = _TRAILING_PUNCT_RE.sub("", text.strip())
    return text


def _clean_question_text(text: str) -> str:
    if not text:
        return text
    text = _SCORE_RE.sub("", text)
    text = _PROMPT_PREFIX_RE.sub("", text)
    text = _LEADING_INDEX_RE.sub("", text.strip())
    text = _TRAILING_PUNCT_RE.sub("", text.strip())
    return text


_SOLVE_SYSTEM_PROMPT = """
你是高中数学解题专家。任务：对用户题目进行完整、严谨、可教学的解题，
并输出结构化的“解题文本”，用于后续转成视频脚本。请遵守以下规则：
必须给出完整解题步骤，条理清晰、行行分明、不可合并成一大段。

输出格式固定为以下六个区块（用【】标记标题）：
【题干复述】
- 用一句话复述题干（保留关键信息）

【分析要点】
- 公式/结论：
  1) ...
  2) ...
- 条件：
  1) ...
  2) ...
- 思路：
  1) ...
  2) ...

【逐步解题】
- 第1步：...（必须只包含一个等式或一个结论）
- 第2步：...（必须只包含一个等式或一个结论）
- ...
- 结论：...（只写最终结论，不要夹带推导）

【逐步解释】
- 对应第1步：说明“为什么这样做”
- 对应第2步：说明“为什么这样做”
- ...
- 对应结论：说明结论含义或必要性

【ModelSpec（物理层）】
- 用题目真实单位（m、s、rad 等）
- 输出：场景几何/接触关系/阶段/事件/每阶段进度函数 s_phys(t)
- 事件示例：到达斜面底端、进入水平面、停下
- 允许使用 key: value 的键值行或条目列表，不要输出 JSON

【DiagramSpec（图示层）】
- 把物理量通过可控映射变成 scene units（画布坐标）
- 给出 world2d 的 x_range / y_range
- 给出比例 u（1 scene unit = u m）与原点/对齐锚点
- 对象使用 diagram 坐标（start/end/pos/points），每个对象必须包含 id（snake_case）
- 只能使用已支持的组件类型（示例）：inclined_plane, rough_surface, smooth_surface,
  horizontal_surface, block, ball, cart, force_arrow, path_polyline, path_arc,
  path_circle, path_projectile, path_parametric, track, text, group
- 力/速度标注必须放在箭头附近（靠近箭头尖或中后段）
- 若出现长距离（如 20m 以上），在 DiagramSpec 中进行压缩或断轴，建议最长线段 <= 12~14 单位
- 若无可视化，写 NO_VISUAL

【MotionSpec（运动层）】
- 用题目真实单位，描述“轨迹 + 阶段 + 姿态规则”
- 输出：body（id/type: block|ball|particle）、track/path 参数、阶段/事件、每阶段 s_phys(t) 或 θ(t)
- 若是圆周/参数曲线，给出中心/半径/角速度/相位，或显式 x(t), y(t)
- 若是分段运动，写清楚每段的区间与连接事件（进入水平/停止等）
- 不要输出 JSON

要求：
1) 若题目含多个小问，请按“(1)(2)(3)”分别给出六个区块。
2) 逐步解题必须是可逐行呈现的表达式或结论，禁止在同一步写多个公式或多段话。
3) 每个小问的步骤数 >= 3（除非题目极短），并且每步只写一句。
4) 逐步解释必须与逐步解题一一对应，解释也要分行，不能合并成一段。
5) 数学公式尽量用 LaTeX 写出（如 \\frac、\\sum），但不要在同一步写多个 $$...$$。
6) 不要输出 JSON（ModelSpec/DiagramSpec 也不要 JSON）。
7) 数学表达必须被 $...$ 包裹（包括数字、变量、算式、等式与常见符号）。
8) frac/sqrt 等必须带反斜杠（如 \\frac、\\sqrt），禁止裸写。
9) DiagramSpec 必须给出明确数值坐标，不要使用“靠近/大约/适当”等模糊描述。
""".strip()

_FORMAT_SYSTEM_PROMPT = """
你是“解题文本 → JSON 脚本”的结构化助手。
你的任务是把输入的解题文本转成严格 JSON，完全符合下面结构：

  {
    "problem_full_text": "完整题面",
    "stem": "题干",
    "questions": [
    {
      "question_text": "小问文本",
      "analysis": {
        "formulas": ["..."],
        "conditions": ["..."],
        "strategy": ["..."]
      },
      "steps": [
        {"line": "每行推导（可含LaTeX）", "subtitle": "解释原因"}
      ],
      "visual": { ... } 或 null,
      "model_spec": { ... } 或 null,
      "diagram_spec": { ... } 或 null,
      "motion_spec": { ... } 或 null
    }
    ]
  }

示例（仅示意格式）：
{
  "problem_full_text": "已知函数 $f(x)=x^2$，求导数。\\n(1) 求 $f'(x)$",
  "stem": "已知函数 $f(x)=x^2$，求导数。",
  "questions": [
    {
      "question_text": "求 $f'(x)$",
      "analysis": {
        "formulas": ["$\\frac{d}{dx}x^n=nx^{n-1}$"],
        "conditions": ["$x\\in\\mathbb{R}$"],
        "strategy": ["套用幂函数求导法则"]
      },
      "steps": [
        {"line": "$f'(x)=2x$", "subtitle": "对 $x^2$ 求导得到 $2x$"}
      ],
      "visual": null,
      "model_spec": null,
      "diagram_spec": null,
      "motion_spec": null
    }
  ]
}
  
规则：
1) 只输出 JSON，不要额外文字。
2) problem_full_text 是完整题干（含小问），必须按“题干 + 每小问各一行”分行展示，使用 \\n 换行；每行尽量不超过20个汉字，过长必须继续用 \\n 断行。
3) stem 是公共题干，不含小问。
4) 若解题文本中包含【ModelSpec】/【DiagramSpec】/【MotionSpec】，请解析后填入对应字段；
   必须优先转成结构化字典，只有完全无法解析时才用 {"text": "..."} 保留原文内容。
DiagramSpec 结构化目标（尽量填齐）：
   {
     "x_range": [xmin, xmax],
     "y_range": [ymin, ymax],
     "u": number 或 "auto",
     "origin": [x0, y0],
     "aligned": "bottom_anchor" 等,
     "objects": [
       {"id": "...", "type": "...", "start": [x,y], "end": [x,y]},
       {"id": "...", "type": "...", "pos": [x,y]},
       {"id": "...", "type": "...", "points": [[x,y], ...]},
       {"id": "...", "type": "...", "center": [x,y], "radius": r}
     ]
   }
   注意字段名：线段/箭头用 start/end；圆用 center/radius；禁止 p1/p2、pos/r 这种非标准键。
   组件类型必须来自允许列表：inclined_plane, rough_surface, smooth_surface, horizontal_surface,
   block, ball, cart, force_arrow, arrow, line, polyline, path_polyline, path_arc, path_circle, track, text, group。
   若 DiagramSpec 文本里有 objects: 列表，必须解析成 diagram_spec.objects。
   MotionSpec 结构化目标（尽量填齐）：
   {
     "bodies": [{"id": "...", "type": "block|ball|particle", "rotate": "auto|none"}],
     "tracks": [{"id": "...", "type": "track|path_circle|path_arc|path_parametric", "params": {...}}],
     "phases": [
       {"id": "...", "body_id": "...", "track_id": "...",
        "s_phys": "...", "t_range": [t0, t1], "events": ["..."]},
       {"id": "...", "body_id": "...", "track_id": "...",
        "keyframes": [[t, s], ...], "s_mode": "absolute|proportion"}
     ]
   }
4) questions 数组的每个元素是一个小问。
5) analysis.formulas/conditions/strategy 从“分析要点”中提取。
6) steps 的 line 来自“逐步解题”，subtitle 来自“逐步解释”，必须一一对应。
7) 所有数学表达式一律用 $...$ 包裹，便于混排渲染（例如：$\\frac{a}{b}$）。
8) 保留 LaTeX 表达式，避免丢失反斜杠。
9) 所有字符串必须是合法 JSON（反斜杠要双写，例如 JSON 中写 \\frac、\\times、\\sqrt）。
10) line 字段每条只包含一个公式或一个结论，不要在同一条里写多个公式。
11) 禁止输出裸露的 frac/sqrt 等命令，必须带反斜杠（如 \\frac、\\sqrt）。
12) 任何数学表达必须被 $...$ 包裹（包括数字、变量、算式、等式与常见符号）。
13) 以下命令必须带反斜杠（示例仅列常见项）：frac、sqrt、sum、prod、int、lim、ln、log、sin、cos、tan、cot、sec、csc、
    arcsin、arccos、arctan、min、max、arg、deg、cdot、times、div、pm、mp、le、ge、neq、approx、sim、propto、in、subset、
    subseteq、cup、cap、setminus、forall、exists、therefore、because、infty、partial、nabla、hbar、left、right、vec、
    overrightarrow、hat、bar、tilde、dot、ddot、mathrm、mathbf，以及所有常见希腊字母（如 alpha、beta、gamma、theta、lambda、mu 等）。
14) 若存在 DiagramSpec，则优先由 diagram_spec 直接生成 visual：
    visual = {"type":"world2d","x_range":diagram_spec.x_range,"y_range":diagram_spec.y_range,
              "children":diagram_spec.objects,"diagram_spec":diagram_spec}
    若 diagram_spec.objects 为空或缺失，则 visual=null。
15) 从【绘图规划】生成 visual：若为 NO_VISUAL，则 visual=null。
16) visual 必须是 world2d 坐标级结构，包含 type/x_range/y_range/children；children 只用已支持组件并带 id 与明确坐标字段。
""".strip()

_VISUAL_SYSTEM_PROMPT = """
你是“解题 JSON → 视觉 JSON”的结构化助手。
输入是一份题目规划 JSON（包含 stem / questions / analysis / steps，可选 visual_hints / model_spec / diagram_spec / motion_spec）。
你的任务是为每个小问生成一个可绘制的 visual 描述，只允许使用已支持的类型。

只输出 JSON，格式固定为：
{
  "visuals": [
    { ... },   // 对应第1小问，若不画图则写 null
    { ... }    // 对应第2小问
  ]
}

可用 visual 类型（仅允许以下几种）：
1) inclined_plane（斜面+滑块+受力箭头）
   {
     "type": "inclined_plane",
     "angle": 30,              // 斜面角度（度）
     "length_scale": 1.0,      // 斜面长度比例（0.4~1.2）
     "block_pos": 0.6,         // 滑块位置（0~1，沿斜面）
     "forces": [
       {"dir": "up_slope", "label": "F"},
       {"dir": "down_slope", "label": "G_∥"},
       {"dir": "normal", "label": "N"}
     ]
   }

2) objects（通用对象列表，适合简单示意）
   {
     "type": "objects",
     "objects": [
       {"type": "plane", "angle": 30, "length": 1.0},
       {"type": "block", "pos": 0.6},
       {"type": "force", "dir": "up_slope", "label": "F"}
     ]
   }

3) component / components（仅用于预览组件，不用于解题图）

可用 force 方向（dir）：up_slope / down_slope / normal / up / down / left / right。

ID 命名规则（非常重要）：
1) 需要后续动画的对象必须写 id（如 block、ball、cart、track/path）。
2) 使用小写 snake_case，仅含字母/数字/下划线，必须以字母开头。
3) 同一小问内 id 唯一，避免重名。
4) 建议约定后缀：block/ball/cart/track/path/arc/line 等（例如 block_1、track_arc）。

规则：
1) 只输出 JSON，不要附加解释文字或 Markdown。
2) visuals 数组长度必须与 questions 数量一致，不画图就写 null。
3) 只用允许的字段与类型；不确定就返回 null。
4) 如果提供了 visual_hints，优先参考其内容。
5) 若输入包含 diagram_spec，请优先采用其 x_range/y_range/u/origin 等设置；
   推荐做法：将 diagram_spec 原样放入 visual 中（visual.diagram_spec），并在其坐标体系内画图。
6) 若输入包含 model_spec / motion_spec，可据此生成轨道/接触/阶段信息，避免凭空设定几何。
""".strip()


_VISUAL_SEQUENCE_SYSTEM_PROMPT = """
你是“解题步骤 → 图形变换”的结构化助手。
输入是题目规划 JSON（含 question_text、steps，可包含 visual / visual_ids / visual_hints / model_spec / diagram_spec / motion_spec）。
你的任务：为每个步骤生成对应的 visual_transform（可为 null）。
务必严格输出 JSON，不要输出任何说明、Markdown、或多余文字。

输出格式固定为：
{
  "questions": [
    {
      "question_index": 0,
      "visual_transforms": [
        [{"action": "show", "target_id": "block", "params": {}, "duration": 0.2}],
        [{"action": "follow_path", "target_id": "block", "params": {"path_id": "track", "reverse": false}, "duration": 1.2}],
        null
      ]
    }
  ]
}

核心约束（必须遵守）：
1) visual_transforms 的长度必须等于 steps 的长度。
2) 每个 step 输出一个“变换列表”或 null。
3) action 必须使用小写且属于允许集合：
   move / shift / rotate / scale / color / opacity / show / hide / remove /
   follow_path / move_along_path / follow_path_segment / follow_path_rotate /
   trajectory_trace / trace / trace_start / ghost / marker
4) target_id 必须来自提供的 visual_ids（若给了 visual_ids），否则该条变换输出 null。
5) path_id 必须来自 visual_ids，且应指向“路径/轨道类对象”，不要随意填。
6) id 必须是小写 snake_case（字母/数字/下划线，且以字母开头），禁止中文/空格。
7) 若没有 visual 或 visual_ids 为空，则整题 visual_transforms 全部为 null。
8) params 必须是对象；不确定就用 {}。
9) duration 取 0.2 ~ 0.6 秒；移动/路径可用 0.6~2.0 秒。
10) 只生成与步骤内容直接相关的动作，保持“可视化与步骤同步”；无可视变化则输出 null。
11) 不要创造新物体、不要写新 id、不要改 visual 本体结构，只做“变换”。

各动作的参数约定：
- move: {"x": float, "y": float, "z": float?}
- shift: {"x": float, "y": float, "z": float?}
- rotate: {"angle": float}  // 度数
- scale: {"scale": float}
- color: {"color": "#RRGGBB"}
- opacity: {"opacity": 0~1}
- show/hide/remove: params 为空 {}
- follow_path / move_along_path:
    params: {"path_id": "轨迹对象id", "reverse": bool?}
- follow_path_segment:
    params: {"path_id": "...", "start": 0~1, "end": 0~1}
- follow_path_rotate:
    params: {"path_id": "...", "start": 0~1, "end": 0~1, "offset": "auto" 或 数值}
- trajectory_trace:
    params: {"id": "trace_id", "stroke_width": number?, "opacity": 0~1?}
- ghost:
    params: {"path_id": "...", "count": int?, "opacity_start": 0~1?, "opacity_end": 0~1?}
- marker:
    params: {"path_id": "...", "count": int?, "radius": number?}

常见映射（参考，但不要生编）：
- “出现/显示/画出” → show
- “消失/去掉” → hide（必要时 remove）
- “沿轨道/沿曲线运动/滑动” → follow_path 或 follow_path_rotate（需 path_id）
- “只走一段轨迹” → follow_path_segment（start/end）
- “改变颜色/高亮/弱化” → color / opacity
- “留痕/轨迹” → trajectory_trace
- “残影/多位置” → ghost
- “标注关键点/时刻” → marker

示例（仅示意格式，不要照抄）：
示例A：
steps = ["建立示意", "滑块沿轨道运动", "显示轨迹"]
visual_ids = ["block_1", "track_arc"]
visual_transforms = [
  [{"action": "show", "target_id": "block_1", "params": {}, "duration": 0.2}],
  [{"action": "follow_path_rotate", "target_id": "block_1", "params": {"path_id": "track_arc", "offset": "auto"}, "duration": 1.2}],
  [{"action": "trajectory_trace", "target_id": "block_1", "params": {"stroke_width": 2.0, "opacity": 0.8}, "duration": 0.2}]
]

示例B：
steps = ["出现并沿轨道运动", "无可视变化"]
visual_ids = ["cart_1", "track_line"]
visual_transforms = [
  [
    {"action": "show", "target_id": "cart_1", "params": {}, "duration": 0.2},
    {"action": "follow_path", "target_id": "cart_1", "params": {"path_id": "track_line"}, "duration": 1.0}
  ],
  null
]

生成策略：
- 优先参考 visual_hints 与 steps 文本；不要凭空创造新物体。
- 若同一步包含“出现某物体并运动”，可在同一步输出多个动作（先 show 再 follow_path）。
- 若目标/路径不明确，输出 null（不要胡写）。
- 如果 motion_spec 指定 body_type=ball/particle，则 rotate 使用 none（不旋转）。
""".strip()

def _prepare_visual_sequence_input(plan: Dict[str, Any], *, solution_text: Optional[str] = None) -> Dict[str, Any]:
    """
    准备visual_sequence的输入数据
    提取问题和步骤信息，去掉不必要的字段
    
    :param plan: 完整的规划字典
    :param solution_text: 可选的原始解题文本
    :return: 简化后的输入字典
    """
    questions = []
    for q in plan.get("questions", []):
        if not isinstance(q, dict):
            continue
        steps = []
        for step in q.get("steps", []):
            if isinstance(step, dict):
                steps.append({
                    "line": str(step.get("line", "")),
                    "subtitle": str(step.get("subtitle", ""))
                })
        
        visual = q.get("visual")
        visual_ids = _collect_visual_ids(visual) if isinstance(visual, dict) else []
        questions.append({
            "question_index": len(questions),
            "question_text": str(q.get("question_text", "")),
            "analysis": q.get("analysis", {}),
            "steps": steps,
            "visual": visual,
            "visual_ids": visual_ids,
            "model_spec": q.get("model_spec"),
            "diagram_spec": q.get("diagram_spec"),
            "motion_spec": q.get("motion_spec"),
        })
    
    out = {
        "stem": str(plan.get("stem", "")),
        "questions": questions
    }
    
    # 如果有visual_hints，添加到输入中
    if solution_text:
        hints = _extract_visual_hints(solution_text)
        if hints:
            out["visual_hints"] = hints
    
    return out


def _collect_visual_ids(visual: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(visual, dict):
        return []
    ids: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            obj_id = node.get("id")
            if isinstance(obj_id, str) and obj_id.strip():
                ids.append(obj_id.strip())
            for child in node.get("children", []) or []:
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(visual)
    seen = set()
    ordered: List[str] = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


_ALLOWED_VISUAL_ACTIONS = {
    "move",
    "shift",
    "rotate",
    "scale",
    "color",
    "opacity",
    "show",
    "hide",
    "remove",
    "follow_path",
    "move_along_path",
    "follow_path_segment",
    "follow_path_rotate",
    "trajectory_trace",
    "trace",
    "trace_start",
    "ghost",
    "marker",
}


def _sanitize_visual_transforms(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    questions = plan_dict.get("questions", [])
    if not isinstance(questions, list):
        return plan_dict
    for q in questions:
        if not isinstance(q, dict):
            continue
        visual_ids = set(_collect_visual_ids(q.get("visual")))
        steps = q.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            transforms = step.get("visual_transform")
            if not isinstance(transforms, list):
                continue
            cleaned = []
            for t in transforms:
                if not isinstance(t, dict):
                    continue
                action = str(t.get("action", "")).strip().lower()
                target_id = str(t.get("target_id", "")).strip()
                if not action or action not in _ALLOWED_VISUAL_ACTIONS:
                    continue
                if not target_id:
                    continue
                if visual_ids and target_id not in visual_ids:
                    continue
                params = t.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                try:
                    duration = float(t.get("duration", 0.3))
                except Exception:
                    duration = 0.3
                cleaned.append(
                    {
                        "action": action,
                        "target_id": target_id,
                        "params": params,
                        "duration": duration,
                    }
                )
            if cleaned:
                step["visual_transform"] = cleaned
            else:
                step.pop("visual_transform", None)
    return plan_dict
