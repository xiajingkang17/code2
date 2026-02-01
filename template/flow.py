from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import math
from manim import (
    DEGREES,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Circle,
    FadeIn,
    FadeOut,
    MoveAlongPath,
    Scene,
    Transform,
    TracedPath,
    UpdateFromAlphaFunc,
    VGroup,
    Rectangle,
    Mobject,
    config,
)

from layout import (
    AnalysisPanel,
    Constraints,
    PinnedHeader,
    SolutionLine,
    Subtitle,
    Theme,
    clear_text_cache,
    apply_overrides,
    compute_metrics,
    decide_layout,
    fit_text_to_box,
    make_full_problem,
    make_text_mobject,
)
from plan import ProblemPlan
from plan.schema import StepVisual
from layout.text_fit import wrap_text_to_char_limit, wrap_text_to_width
from .visuals import build_visual_with_dict, apply_visual_transform


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_audio_manifest() -> dict[tuple[int, int], dict[str, object]]:
    manifest = os.environ.get("AUDIO_MANIFEST")
    if not manifest:
        return {}
    path = Path(manifest)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    base_dir = path.parent
    if isinstance(data, dict) and data.get("base_dir"):
        base_dir = (path.parent / str(data["base_dir"])).resolve()
    entries = data.get("entries", []) if isinstance(data, dict) else []
    audio_map: dict[tuple[int, int], dict[str, object]] = {}
    for entry in entries:
        try:
            qi = int(entry.get("q"))
            si = int(entry.get("s"))
            rel = entry.get("path")
            duration = float(entry.get("duration", 0.0))
        except Exception:
            continue
        if not rel:
            continue
        full_path = Path(rel)
        if not full_path.is_absolute():
            full_path = (base_dir / full_path).resolve()
        audio_map[(qi, si)] = {"path": full_path, "duration": duration}
    return audio_map


def _analysis_weight(items: list[str]) -> float:
    if not items:
        return 1.0
    total_len = sum(len(str(x)) for x in items)
    avg_len = total_len / max(1, len(items))
    # Weight by count + a little by average length.
    return max(1.0, len(items) + avg_len / 20.0)


def _available_analysis_height(frame_h: float, header: Optional[VGroup], constraints: Constraints) -> float:
    top_y = frame_h / 2 - constraints.safe_top
    bottom_y = -frame_h / 2 + constraints.safe_bottom
    header_bottom = header.get_bottom()[1] if header is not None else top_y
    return max(0.0, header_bottom - constraints.min_margin - bottom_y)


def _compose_questions(plan: ProblemPlan) -> list[str]:
    lines: list[str] = []
    for idx, q in enumerate(plan.questions, start=1):
        text = (q.question_text or "").strip()
        if not text:
            continue
        if not text.startswith("("):
            text = f"({idx}) {text}"
        lines.append(text)
    return lines


def _build_full_problem_columns(
    plan: ProblemPlan,
    frame_w: float,
    frame_h: float,
    font: str,
    font_size: float,
    left_ratio: float,
    gap: float,
    max_chars_left: int,
    max_chars_right: int,
    scale_up: float,
    max_height_ratio: float,
) -> VGroup:
    stem = (plan.stem or "").strip()
    questions = _compose_questions(plan)
    left_text = wrap_text_to_char_limit(stem, max_chars=max_chars_left)
    right_text = wrap_text_to_char_limit("\n".join(questions), max_chars=max_chars_right)

    left = make_text_mobject(left_text, font=font, font_size=font_size)
    right = make_text_mobject(right_text, font=font, font_size=font_size)

    max_h = frame_h * max_height_ratio
    total_w = frame_w * 0.9
    max_w_left = total_w * left_ratio
    max_w_right = total_w * (1.0 - left_ratio) - gap
    left_scale = min(max_w_left / left.width, max_h / left.height)
    right_scale = min(max_w_right / right.width, max_h / right.height)
    scale = min(left_scale, right_scale)
    if scale > 1.0:
        scale = min(scale, scale_up)
    if scale != 1.0:
        left.scale(scale)
        right.scale(scale)

    group = VGroup(left, right).arrange(RIGHT, aligned_edge=UP, buff=gap)
    return group


@dataclass
class LayoutConfig:
    header_width_ratio: float = 0.55
    header_height_ratio: float = 0.24
    analysis_width_ratio: float = 0.52
    analysis_height_ratio: float = 0.36
    solution_width_ratio: float = 0.85
    solution_height_ratio: float = 0.45
    visual_width_ratio: float = 0.5
    steps_width_ratio: float = 0.55
    subtitle_width_ratio: float = 0.9
    subtitle_height_ratio: float = 0.1
    theme: Theme = field(default_factory=Theme)
    constraints: Constraints = field(default_factory=Constraints)
    layout_debug: bool = False
    line_anim_time: float = 0.25
    subtitle_anim_time: float = 0.25
    audio_lead: float = 0.0
    audio_tail: float = 0.12
    audio_min_wait: float = 0.4
    anim_min_scale: float = 0.5
    anim_max_scale: float = 1.5
    full_problem_layout: str = "columns"
    full_problem_font_size: float = 36.0
    full_problem_left_ratio: float = 0.5
    full_problem_gap: float = 0.6
    full_problem_top_margin: float = 0.55
    full_problem_max_chars_left: int = 20
    full_problem_max_chars_right: int = 18
    full_problem_scale_up: float = 1.3
    full_problem_max_height_ratio: float = 0.33


class ProblemSceneBase(Scene):
    def __init__(self, renderer=None, layout: Optional[LayoutConfig] = None, **kwargs) -> None:
        # Manim instantiates SceneClass(renderer) positionally; accept it here.
        super().__init__(renderer=renderer, **kwargs)
        config.background_color = "#000000"
        self.layout = layout or LayoutConfig()
        self._pinned_header: Optional[PinnedHeader] = None
        self._analysis_group: Optional[VGroup] = None
        self._solution_group = VGroup()
        self._subtitle: Optional[Subtitle] = None
        self._visual_group: Optional[VGroup] = None
        self._visual_mobject_dict: Dict[str, Mobject] = {}  # 存储 visual 中各个 id 对应的 mobject
        self._debug_group: Optional[VGroup] = None
        self._audio_map = _load_audio_manifest()
        self.layout.line_anim_time = _read_float_env("LINE_ANIM_TIME", self.layout.line_anim_time)
        self.layout.subtitle_anim_time = _read_float_env("SUBTITLE_ANIM_TIME", self.layout.subtitle_anim_time)
        self.layout.audio_lead = _read_float_env("AUDIO_LEAD", self.layout.audio_lead)
        self.layout.audio_tail = _read_float_env("AUDIO_TAIL", self.layout.audio_tail)
        self.layout.audio_min_wait = _read_float_env("AUDIO_MIN_WAIT", self.layout.audio_min_wait)
        self.layout.anim_min_scale = _read_float_env("ANIM_MIN_SCALE", self.layout.anim_min_scale)
        self.layout.anim_max_scale = _read_float_env("ANIM_MAX_SCALE", self.layout.anim_max_scale)

    def _frame_size(self) -> tuple[float, float]:
        # OpenGLCamera may not expose frame_width/height; fall back to global config.
        try:
            return self.camera.frame_width, self.camera.frame_height
        except AttributeError:
            return config.frame_width, config.frame_height

    def play_problem(self, plan: ProblemPlan) -> None:
        self.show_full_problem(plan)
        for qi, q in enumerate(plan.questions, start=1):
            self.pin_header(plan.stem, q.question_text, q.layout_overrides)
            self._hide_visual()
            self.show_analysis(q)
            self.clear_analysis()
            self.show_visual(q)
            self.write_steps(q, qi)
            self.transition_to_next_question()
        clear_text_cache()

    def show_full_problem(self, plan: ProblemPlan) -> None:
        frame_w, frame_h = self._frame_size()
        if self.layout.full_problem_layout == "columns" and plan.stem and plan.questions:
            full = _build_full_problem_columns(
                plan,
                frame_w,
                frame_h,
                font=self.layout.theme.font,
                font_size=self.layout.full_problem_font_size,
                left_ratio=self.layout.full_problem_left_ratio,
                gap=self.layout.full_problem_gap,
                max_chars_left=self.layout.full_problem_max_chars_left,
                max_chars_right=self.layout.full_problem_max_chars_right,
                scale_up=self.layout.full_problem_scale_up,
                max_height_ratio=self.layout.full_problem_max_height_ratio,
            )
        else:
            raw_text = plan.problem_full_text
            if "\\n" not in raw_text and "\n" not in raw_text:
                raw_text = wrap_text_to_width(
                    raw_text,
                    max_width=frame_w * 0.9,
                    font=self.layout.theme.font,
                    font_size=self.layout.full_problem_font_size,
                )
            full = make_full_problem(
                raw_text,
                max_width=frame_w * 0.9,
                max_height=frame_h * 0.9,
                font=self.layout.theme.font,
            )
            if full.width > 0 and full.height > 0:
                max_w = frame_w * 0.9
                max_h = frame_h * self.layout.full_problem_max_height_ratio
                scale = min(max_w / full.width, max_h / full.height)
                if scale > 1.0:
                    full.scale(min(scale, self.layout.full_problem_scale_up))
        top_y = frame_h / 2 - self.layout.full_problem_top_margin
        full.move_to(UP * (top_y - full.height / 2))
        self.play(FadeIn(full))
        self.wait(2)
        self.play(FadeOut(full))

    def _effective_layout(self, q) -> tuple[Theme, Constraints]:
        return apply_overrides(self.layout.theme, self.layout.constraints, getattr(q, "layout_overrides", None))

    def _clear_debug(self) -> None:
        if self._debug_group is not None:
            self.remove(self._debug_group)
            self._debug_group = None

    def _draw_debug_frame(self, frame_w: float, frame_h: float, constraints: Constraints) -> None:
        if not self.layout.layout_debug:
            return
        self._clear_debug()
        safe_h = frame_h - constraints.safe_top - constraints.safe_bottom
        safe = Rectangle(width=frame_w - constraints.min_margin * 2, height=safe_h)
        safe.shift(UP * (constraints.safe_bottom - constraints.safe_top) / 2)
        safe.set_stroke(width=2, color="#2a6fdb")
        safe.set_fill(opacity=0)
        frame = Rectangle(width=frame_w, height=frame_h)
        frame.set_stroke(width=1, color="#888888")
        frame.set_fill(opacity=0)
        self._debug_group = VGroup(frame, safe)
        self.add(self._debug_group)

    def pin_header(self, stem: str, question_text: str, overrides=None) -> None:
        frame_w, frame_h = self._frame_size()
        theme, constraints = apply_overrides(self.layout.theme, self.layout.constraints, overrides)
        self._draw_debug_frame(frame_w, frame_h, constraints)
        max_width_ratio = min(self.layout.header_width_ratio, constraints.max_width_ratio)
        header = PinnedHeader(
            stem,
            question_text,
            max_width=frame_w * max_width_ratio,
            max_height=frame_h * self.layout.header_height_ratio,
            font=theme.font,
            theme=theme,
            constraints=constraints,
        )
        header.to_corner(UP + LEFT, buff=constraints.min_margin)
        header.shift(DOWN * constraints.safe_top * 0.5)
        if self._pinned_header is None:
            self._pinned_header = header
            self.play(FadeIn(header))
        else:
            self.play(Transform(self._pinned_header, header))

    def show_analysis(self, q) -> None:
        frame_w, frame_h = self._frame_size()
        theme, constraints = self._effective_layout(q)
        max_width_ratio = min(self.layout.analysis_width_ratio, constraints.max_width_ratio)
        sections: list[tuple[str, list[str]]] = []
        if q.analysis.formulas:
            sections.append(("Formulas", list(q.analysis.formulas)))
        if q.analysis.conditions:
            sections.append(("Conditions", list(q.analysis.conditions)))
        if q.analysis.strategy:
            sections.append(("Strategy", list(q.analysis.strategy)))
        panels: list[AnalysisPanel] = []
        if not sections:
            return

        available_h = _available_analysis_height(frame_h, self._pinned_header, constraints)
        total_weight = sum(_analysis_weight(items) for _, items in sections) or 1.0
        base_cap = frame_h * self.layout.analysis_height_ratio
        for title, items in sections:
            weight = _analysis_weight(items)
            max_height = base_cap
            if available_h > 0:
                max_height = min(base_cap, available_h * (weight / total_weight))
            panels.append(
                AnalysisPanel(
                    title,
                    items,
                    max_width=frame_w * max_width_ratio,
                    max_height=max_height,
                    font=theme.font,
                    theme=theme,
                    constraints=constraints,
                )
            )

        group = VGroup(*panels).arrange(DOWN, aligned_edge=LEFT, buff=0.3 * theme.line_spacing)
        if available_h > 0 and group.height > available_h:
            group.scale_to_fit_height(available_h)
        group.next_to(self._pinned_header, DOWN, buff=constraints.min_margin, aligned_edge=LEFT)
        self._analysis_group = group
        self.play(FadeIn(group))
        self.wait(0.6)

    def clear_analysis(self) -> None:
        if self._analysis_group is None:
            return
        self.play(FadeOut(self._analysis_group))
        self._analysis_group = None

    def _hide_visual(self) -> None:
        if self._visual_group is None:
            return
        self.play(FadeOut(self._visual_group))
        self._visual_group = None

    def _visual_area(self, frame_w: float, frame_h: float, constraints: Constraints) -> tuple[float, float, float, float]:
        left = -frame_w / 2 + constraints.min_margin
        max_width_ratio = min(
            self.layout.visual_width_ratio,
            self.layout.header_width_ratio,
            constraints.max_width_ratio,
        )
        width = frame_w * max_width_ratio
        right = left + width
        top = frame_h / 2 - constraints.safe_top
        if self._pinned_header is not None:
            top = self._pinned_header.get_bottom()[1] - constraints.min_margin
        bottom = -frame_h / 2 + constraints.safe_bottom
        return left, right, top, bottom

    def show_visual(self, q) -> None:
        spec = getattr(q, "visual", None)
        if spec is None:
            return
        if isinstance(spec, dict) and spec.get("disabled"):
            return
        frame_w, frame_h = self._frame_size()
        theme, constraints = self._effective_layout(q)
        left, right, top, bottom = self._visual_area(frame_w, frame_h, constraints)
        width = max(0.1, right - left)
        height = max(0.1, top - bottom)

        result, id_map = build_visual_with_dict(spec, width, height, theme=theme)
        if result.warning in {"empty_visual", "missing_type"}:
            return
        # 保存步骤级变换所需的 id -> mobject 映射
        self._visual_mobject_dict = id_map or {}
        visual = VGroup(result.mobject)
        visual.move_to(((left + right) / 2) * RIGHT + ((top + bottom) / 2) * UP)

        if self._visual_group is None:
            self._visual_group = visual
            self.add(visual)
        else:
            # 直接替换，避免 FadeIn 覆盖子对象的可见性状态
            self.remove(self._visual_group)
            self._visual_group = visual
            self.add(visual)

    def write_steps(self, q, q_index: int) -> None:
        frame_w, frame_h = self._frame_size()
        theme, constraints = self._effective_layout(q)
        max_width_ratio = min(self.layout.steps_width_ratio, constraints.max_width_ratio)
        max_width = frame_w * max_width_ratio
        texts = [q.question_text]
        texts.extend(q.analysis.formulas)
        texts.extend(q.analysis.conditions)
        texts.extend(q.analysis.strategy)
        texts.extend([s.line for s in q.steps])
        texts.extend([s.subtitle for s in q.steps])
        metrics = compute_metrics(texts, line_count=len(q.steps))
        decision = decide_layout(metrics, constraints)
        theme = Theme(
            font=theme.font,
            font_size_scale=theme.font_size_scale * decision.font_scale,
            line_spacing=theme.line_spacing,
            panel_padding=theme.panel_padding,
            subtitle_bg_opacity=theme.subtitle_bg_opacity,
            accent_color=theme.accent_color,
        )
        max_lines = decision.max_lines

        if self._solution_group:
            self.play(FadeOut(self._solution_group))
            self._solution_group = VGroup()

        self.add(self._solution_group)

        for si, step in enumerate(q.steps, start=1):
            line = SolutionLine(step.line, max_width=max_width, font=theme.font, theme=theme, constraints=constraints)
            self._solution_group.add(line)
            if len(self._solution_group) > max_lines:
                self._solution_group.remove(self._solution_group[0])
            self._solution_group.arrange(DOWN, aligned_edge=LEFT, buff=0.25 * theme.line_spacing)
            self._solution_group.to_edge(RIGHT, buff=constraints.min_margin)
            self._solution_group.shift(DOWN * 0.2 + DOWN * constraints.safe_top * 0.15)

            audio = self._audio_map.get((q_index, si))
            base_line = self.layout.line_anim_time
            base_sub = self.layout.subtitle_anim_time
            line_time = base_line
            sub_time = base_sub
            audio_lead = max(0.0, self.layout.audio_lead)

            if audio:
                target = max(self.layout.audio_min_wait, audio["duration"] + self.layout.audio_tail)
                non_wait = base_line + base_sub + audio_lead
                if non_wait > 0:
                    if target < non_wait:
                        scale = max(self.layout.anim_min_scale, target / non_wait)
                        line_time = base_line * scale
                        sub_time = base_sub * scale
                        non_wait = line_time + sub_time + audio_lead
                    elif target > non_wait:
                        scale = min(self.layout.anim_max_scale, target / non_wait)
                        if scale > 1:
                            line_time = base_line * scale
                            sub_time = base_sub * scale
                            non_wait = line_time + sub_time + audio_lead
                wait_time = max(self.layout.audio_min_wait, target - non_wait)
            else:
                wait_time = self.layout.audio_min_wait

            # 新增：应用步骤级visual变换
            if hasattr(step, 'visual_transform') and step.visual_transform:
                for transform in step.visual_transform:
                    self._apply_step_visual_transform(transform, run_time=transform.duration)

            self.play(FadeIn(line), run_time=line_time)
            self.update_subtitle(step.subtitle, theme=theme, constraints=constraints, run_time=sub_time)
            if audio and audio["path"].exists():
                if audio_lead > 0:
                    self.wait(audio_lead)
                self.add_sound(str(audio["path"]))
            self.wait(wait_time)

    def update_subtitle(
        self,
        text: str,
        theme: Optional[Theme] = None,
        constraints: Optional[Constraints] = None,
        run_time: Optional[float] = None,
    ) -> None:
        frame_w, _ = self._frame_size()
        theme = theme or self.layout.theme
        constraints = constraints or self.layout.constraints
        max_width_ratio = min(self.layout.subtitle_width_ratio, constraints.max_width_ratio)
        subtitle = Subtitle(
            text,
            max_width=frame_w * max_width_ratio,
            font=theme.font,
            theme=theme,
            constraints=constraints,
        )
        subtitle.to_edge(DOWN, buff=constraints.min_margin)
        subtitle.shift(UP * constraints.safe_bottom * 0.5)
        if self._subtitle is None:
            self._subtitle = subtitle
            self.play(FadeIn(subtitle), run_time=run_time)
        else:
            self.play(Transform(self._subtitle, subtitle), run_time=run_time)

    def _apply_step_visual_transform(self, transform: StepVisual, run_time: float = 0.3) -> None:
        """
        应用单个步骤级的图形变换
        
        :param transform: StepVisual对象，包含要应用的变换信息
        :param run_time: 动画运行时长（秒），如果为None则使用transform中的duration
        """
        if not self._visual_group or not self._visual_mobject_dict:
            return
        
        target_id = transform.target_id
        if target_id not in self._visual_mobject_dict:
            return
        
        mobj = self._visual_mobject_dict[target_id]
        action = transform.action.strip().lower()
        params = transform.params or {}
        animation_time = float(getattr(transform, "duration", run_time) or run_time)

        if action == "remove":
            self.play(FadeOut(mobj), run_time=animation_time)
            self._remove_mobject_from_map(mobj)
            return

        anim = None
        if action == "move":
            cx, cy, cz = mobj.get_center()
            x = self._safe_float(params.get("x"), default=cx)
            y = self._safe_float(params.get("y"), default=cy)
            z = self._safe_float(params.get("z"), default=cz)
            anim = mobj.animate.move_to([x, y, z])
        elif action == "shift":
            x = self._safe_float(params.get("x"), default=0.0)
            y = self._safe_float(params.get("y"), default=0.0)
            z = self._safe_float(params.get("z"), default=0.0)
            anim = mobj.animate.shift([x, y, z])
        elif action == "rotate":
            angle_deg = self._safe_float(params.get("angle"), default=0.0)
            if angle_deg != 0:
                anim = mobj.animate.rotate(angle_deg * DEGREES)
        elif action == "scale":
            scale_factor = self._safe_float(params.get("scale"), default=1.0)
            if scale_factor > 0 and scale_factor != 1.0:
                anim = mobj.animate.scale(scale_factor)
        elif action in {"follow_path", "move_along_path", "follow_path_segment", "follow_path_rotate"}:
            path_id = str(params.get("path_id", "")).strip()
            anchor = None
            if isinstance(mobj, VGroup):
                anchor_id = params.get("anchor_id")
                if anchor_id:
                    anchor = self._visual_mobject_dict.get(str(anchor_id))
                if anchor is None:
                    anchor = self._visual_mobject_dict.get(f"{target_id}_anchor")
                if anchor is None:
                    anchor = self._visual_mobject_dict.get(f"{target_id}_shape")
            reverse = bool(params.get("reverse", False))
            raw_start = params.get("start", 0.0)
            raw_end = params.get("end", 1.0)
            start = self._safe_float(raw_start, default=0.0)
            end = self._safe_float(raw_end, default=1.0)
            angle_offset_deg = self._safe_float(params.get("angle_offset_deg"), default=0.0)
            angle_offset = params.get("angle_offset")
            if angle_offset is not None:
                angle_offset_val = self._safe_float(angle_offset, default=0.0)
            else:
                angle_offset_val = angle_offset_deg * DEGREES
            angle_mode = str(params.get("angle_mode", "absolute")).strip().lower()
            absolute_rotate = angle_mode != "relative"
            rotate_param = params.get("rotate")
            body_type = str(params.get("body_type", "")).strip().lower()
            rotate_enabled = True
            if isinstance(rotate_param, str):
                rotate_enabled = rotate_param.strip().lower() not in {"none", "false", "off", "no"}
            elif rotate_param is False:
                rotate_enabled = False
            if body_type in {"ball", "particle"}:
                rotate_enabled = False

            if not path_id:
                return
            path = self._resolve_path_mobject(path_id)
            if path is None:
                return
            if action == "follow_path_rotate":
                if not rotate_enabled:
                    anim = self._follow_path_segment(mobj, path, anchor=anchor, start=start, end=end, reverse=reverse)
                    # fall through to play animation
                offset_raw = params.get("offset")
                offset_auto = bool(params.get("offset_auto", False))
                offset_val = None
                if isinstance(offset_raw, str) and offset_raw.strip().lower() == "auto":
                    offset_val = mobj.height * 0.5
                elif offset_raw is not None:
                    offset_val = self._safe_float(offset_raw, default=0.0)
                elif offset_auto:
                    offset_val = mobj.height * 0.5
                if anchor is not None:
                    if offset_raw is None or (isinstance(offset_raw, str) and offset_raw.strip().lower() == "auto") or offset_auto:
                        offset_val = None
                anim = self._follow_path_with_rotation(
                    mobj,
                    path,
                    anchor=anchor,
                    start=start,
                    end=end,
                    reverse=reverse,
                    angle_offset=angle_offset_val,
                    offset=offset_val,
                    absolute=absolute_rotate,
                )
            elif action == "follow_path_segment":
                anim = self._follow_path_segment(mobj, path, anchor=anchor, start=start, end=end, reverse=reverse)
            else:
                anim = self._follow_path_segment(
                    mobj, path, anchor=anchor, start=start, end=end, reverse=reverse, allow_move_along=True
                )
        elif action in {"trajectory_trace", "trace", "trace_start"}:
            trace_id = str(params.get("id") or params.get("trace_id") or f"trace_{target_id}")
            color = str(params.get("color", "#ffffff"))
            stroke_width = self._safe_float(params.get("stroke_width"), default=2.0)
            opacity = self._safe_float(params.get("opacity"), default=1.0)
            dissipating_time = self._safe_float(params.get("dissipating_time"), default=0.0)
            trace = TracedPath(
                mobj.get_center,
                stroke_color=color,
                stroke_width=stroke_width,
                dissipating_time=max(0.0, dissipating_time),
            )
            setattr(trace, "_stroke_only", True)
            self._set_mobject_opacity(trace, opacity)
            self._attach_visual(trace, trace_id)
            return
        elif action == "ghost":
            path_id = str(params.get("path_id", "")).strip()
            ghost_id = str(params.get("id") or params.get("ghost_id") or f"ghost_{target_id}")
            count = int(params.get("count", 6))
            proportions = params.get("proportions")
            opacity = self._safe_float(params.get("opacity"), default=0.25)
            opacity_start = self._safe_float(params.get("opacity_start"), default=opacity)
            opacity_end = self._safe_float(params.get("opacity_end"), default=opacity_start)
            scale = self._safe_float(params.get("scale"), default=1.0)
            points = params.get("points")
            ghost_group = self._make_ghosts(
                mobj,
                path_id=path_id,
                points=points,
                proportions=proportions,
                count=count,
                opacity_start=opacity_start,
                opacity_end=opacity_end,
                scale=scale,
            )
            if ghost_group is None:
                return
            self._attach_visual(ghost_group, ghost_id)
            return
        elif action == "marker":
            path_id = str(params.get("path_id", "")).strip()
            marker_id = str(params.get("id") or params.get("marker_id") or f"marker_{target_id}")
            count = int(params.get("count", 5))
            proportions = params.get("proportions")
            points = params.get("points")
            radius = self._safe_float(params.get("radius"), default=0.06)
            stroke_width = self._safe_float(params.get("stroke_width"), default=2.0)
            color = str(params.get("color", "#ffffff"))
            marker_group = self._make_markers(
                path_id=path_id,
                points=points,
                proportions=proportions,
                count=count,
                radius=radius,
                stroke_width=stroke_width,
                color=color,
            )
            if marker_group is None:
                return
            self._attach_visual(marker_group, marker_id)
            return
        elif action == "color":
            color = str(params.get("color", "")).strip()
            if color:
                anim = mobj.animate.set_color(color)
        elif action == "opacity":
            opacity = self._safe_float(params.get("opacity"), default=1.0)
            opacity = max(0.0, min(1.0, opacity))
            if self._is_stroke_only(mobj):
                self._set_mobject_visibility(mobj, opacity > 0)
                return
            anim = mobj.animate.set_opacity(opacity)
        elif action == "show":
            self._set_mobject_visibility(mobj, True)
            return
        elif action == "hide":
            self._set_mobject_visibility(mobj, False)
            return

        if anim is not None:
            self.play(anim, run_time=animation_time)
            return

        # 兜底：如果 action 不支持动画，至少保持语义正确
        apply_visual_transform({target_id: mobj}, transform)

    def _safe_float(self, value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _is_auto_value(self, value) -> bool:
        return isinstance(value, str) and value.strip().lower() == "auto"

    def _is_stroke_only(self, mobj: Mobject) -> bool:
        return bool(getattr(mobj, "_stroke_only", False))

    def _set_mobject_visibility(self, mobj: Mobject, visible: bool) -> None:
        # Apply recursively to keep text labels visible while shapes are stroke-only.
        if hasattr(mobj, "submobjects") and mobj.submobjects:
            for sub in mobj.submobjects:
                self._set_mobject_visibility(sub, visible)
            return
        if self._is_stroke_only(mobj):
            try:
                mobj.set_stroke(opacity=1.0 if visible else 0.0)
            except Exception:
                pass
            try:
                mobj.set_fill(opacity=0.0)
            except Exception:
                pass
            return
        try:
            mobj.set_opacity(1.0 if visible else 0.0)
        except Exception:
            pass

    def _set_mobject_opacity(self, mobj: Mobject, opacity: float) -> None:
        if hasattr(mobj, "submobjects") and mobj.submobjects:
            for sub in mobj.submobjects:
                self._set_mobject_opacity(sub, opacity)
            return
        if self._is_stroke_only(mobj):
            try:
                mobj.set_stroke(opacity=opacity)
            except Exception:
                pass
            try:
                mobj.set_fill(opacity=0.0)
            except Exception:
                pass
            return
        try:
            mobj.set_opacity(opacity)
        except Exception:
            pass

    def _remove_mobject_from_map(self, mobj: Mobject) -> None:
        keys = [k for k, v in self._visual_mobject_dict.items() if v is mobj]
        for key in keys:
            del self._visual_mobject_dict[key]

    def _attach_visual(self, mobj: Mobject, obj_id: Optional[str] = None) -> None:
        if self._visual_group is not None:
            self._visual_group.add(mobj)
        else:
            self.add(mobj)
        if obj_id:
            self._visual_mobject_dict[obj_id] = mobj

    def _resolve_path_mobject(self, path_id: str) -> Optional[Mobject]:
        path = self._visual_mobject_dict.get(path_id)
        if path is None:
            return None
        if hasattr(path, "point_from_proportion"):
            try:
                if not path.has_no_points():
                    return path
            except Exception:
                # If the object cannot report points, still try submobjects.
                pass
        if hasattr(path, "submobjects") and path.submobjects:
            for sub in path.submobjects:
                if hasattr(sub, "point_from_proportion"):
                    try:
                        if sub.has_no_points():
                            continue
                    except Exception:
                        pass
                    return sub
        return None

    def _follow_path_segment(
        self,
        mobj: Mobject,
        path: Mobject,
        *,
        anchor: Optional[Mobject] = None,
        start: float,
        end: float,
        reverse: bool = False,
        allow_move_along: bool = False,
    ) -> Optional[UpdateFromAlphaFunc]:
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        if reverse:
            start, end = end, start
        if allow_move_along and anchor is None and abs(start) < 1e-6 and abs(end - 1.0) < 1e-6:
            path_copy = path.copy()
            if reverse and hasattr(path_copy, "reverse_direction"):
                path_copy.reverse_direction()
            return MoveAlongPath(mobj, path_copy)

        def updater(mob: Mobject, alpha: float) -> None:
            t = start + (end - start) * alpha
            t = max(0.0, min(1.0, t))
            target = path.point_from_proportion(t)
            if anchor is not None:
                offset = anchor.get_center() - mob.get_center()
                mob.move_to(target - offset)
            else:
                mob.move_to(target)

        return UpdateFromAlphaFunc(mobj, updater)

    def _follow_path_with_rotation(
        self,
        mobj: Mobject,
        path: Mobject,
        *,
        anchor: Optional[Mobject] = None,
        start: float,
        end: float,
        reverse: bool = False,
        angle_offset: float = 0.0,
        offset: Optional[float] = None,
        absolute: bool = True,
    ) -> Optional[UpdateFromAlphaFunc]:
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        if reverse:
            start, end = end, start
        base = mobj.copy()
        base_angle = self._get_mobject_angle(mobj) if absolute else 0.0
        anchor_offset = None
        if anchor is not None:
            anchor_offset = anchor.get_center() - mobj.get_center()

        def updater(mob: Mobject, alpha: float) -> None:
            t = start + (end - start) * alpha
            t = max(0.0, min(1.0, t))
            pos = path.point_from_proportion(t)
            if offset:
                tangent = self._path_tangent_vector(path, t)
                normal = np.array([-tangent[1], tangent[0], 0.0])
                pos = pos + normal * float(offset)
            mob.become(base.copy())
            if anchor_offset is not None:
                mob.move_to(pos - anchor_offset)
            else:
                mob.move_to(pos)
            tangent = self._path_tangent_vector(path, t)
            angle = float(np.arctan2(tangent[1], tangent[0]))
            rot_point = anchor.get_center() if anchor_offset is not None else mob.get_center()
            mob.rotate(angle + angle_offset - base_angle, about_point=rot_point)

        return UpdateFromAlphaFunc(mobj, updater)

    def _path_tangent_angle(self, path: Mobject, t: float) -> float:
        tangent = self._path_tangent_vector(path, t)
        return float(np.arctan2(tangent[1], tangent[0]))

    def _path_tangent_vector(self, path: Mobject, t: float) -> np.ndarray:
        t0 = max(0.0, min(1.0, t))
        eps = 1e-3
        t1 = min(1.0, t0 + eps)
        if abs(t1 - t0) < 1e-6:
            t1 = max(0.0, t0 - eps)
        p0 = path.point_from_proportion(t0)
        p1 = path.point_from_proportion(t1)
        vec = np.array([float(p1[0] - p0[0]), float(p1[1] - p0[1]), 0.0])
        norm = np.linalg.norm(vec[:2])
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return vec / norm

    def _get_mobject_angle(self, mobj: Mobject) -> float:
        base = getattr(mobj, "_angle_base", None)
        if base is not None:
            try:
                return float(base)
            except Exception:
                return 0.0
        for attr in ("get_angle", "get_rotation"):
            getter = getattr(mobj, attr, None)
            if callable(getter):
                try:
                    return float(getter())
                except Exception:
                    pass
        if hasattr(mobj, "submobjects") and mobj.submobjects:
            for sub in mobj.submobjects:
                for attr in ("get_angle", "get_rotation"):
                    getter = getattr(sub, attr, None)
                    if callable(getter):
                        try:
                            return float(getter())
                        except Exception:
                            pass
        return 0.0

    def _make_markers(
        self,
        *,
        path_id: str,
        points,
        proportions,
        count: int,
        radius: float,
        stroke_width: float,
        color: str,
    ) -> Optional[VGroup]:
        samples = self._resolve_path_points(path_id, points, proportions, count)
        if not samples:
            return None
        group = VGroup()
        for pos in samples:
            circ = Circle(radius=radius)
            circ.move_to(pos)
            circ.set_stroke(color=color, width=stroke_width)
            circ.set_fill(opacity=0)
            setattr(circ, "_stroke_only", True)
            group.add(circ)
        return group

    def _make_ghosts(
        self,
        mobj: Mobject,
        *,
        path_id: str,
        points,
        proportions,
        count: int,
        opacity_start: float,
        opacity_end: float,
        scale: float,
    ) -> Optional[VGroup]:
        samples = self._resolve_path_points(path_id, points, proportions, count)
        if not samples:
            return None
        count = len(samples)
        group = VGroup()
        for idx, pos in enumerate(samples):
            ratio = 0.0 if count <= 1 else idx / (count - 1)
            opacity = opacity_start + (opacity_end - opacity_start) * ratio
            ghost = mobj.copy()
            if scale != 1.0:
                ghost.scale(scale)
            ghost.move_to(pos)
            self._set_mobject_opacity(ghost, opacity)
            group.add(ghost)
        return group

    def _resolve_path_points(self, path_id: str, points, proportions, count: int) -> list[np.ndarray]:
        if points:
            samples = []
            if isinstance(points, (list, tuple)):
                for pt in points:
                    try:
                        x, y = pt[0], pt[1]
                    except Exception:
                        continue
                    samples.append(np.array([float(x), float(y), 0.0]))
            return samples
        path = self._resolve_path_mobject(path_id) if path_id else None
        if path is None:
            return []
        if proportions and isinstance(proportions, (list, tuple)):
            props = [self._safe_float(p, default=0.0) for p in proportions]
        else:
            count = max(1, int(count))
            if count == 1:
                props = [0.5]
            else:
                props = [i / (count - 1) for i in range(count)]
        samples = []
        for t in props:
            t = max(0.0, min(1.0, t))
            samples.append(path.point_from_proportion(t))
        return samples

    def transition_to_next_question(self) -> None:
        if self._subtitle is not None:
            self.play(FadeOut(self._subtitle))
            self._subtitle = None
        if self._solution_group:
            self.play(FadeOut(self._solution_group))
            self._solution_group = VGroup()
        self._visual_mobject_dict = {}  # 清空mobject字典
        self._clear_debug()
