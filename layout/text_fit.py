from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Optional

# 导入Manim核心组件：方向常量、基础图形对象、文本/LaTeX渲染组件
from manim import LEFT, RIGHT, UP, DOWN, VGroup, Mobject, Text, MathTex


def _contains_cjk(text: str) -> bool:
    # 简单判断是否包含中日韩字符，避免整段中文被误判为 LaTeX
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            return True
    return False


def is_latex(text: str) -> bool:
    """
    判断文本是否包含LaTeX语法（用于数学公式/符号渲染）
    :param text: 待判断的文本字符串
    :return: 包含LaTeX语法返回True，否则返回False
    判定依据：文本中包含\（转义符）、^（上标）、_（下标）、$（LaTeX标识符）任一字符
    注意：若包含中文字符，则默认按普通文本处理
    """
    if _contains_cjk(text):
        return False
    return "\\" in text or "^" in text or "_" in text or "$" in text


def _normalize_latex(text: str) -> str:
    # MathTex 不需要 $...$ 包裹，去掉所有 $ 以减少报错
    return text.replace("$", "").strip()


def _split_inline_math(text: str) -> list[tuple[str, bool]]:
    # 按 $...$ 分割为 [("文本", False), ("公式", True), ...]
    parts: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_math = False
    for ch in text:
        if ch == "$":
            segment = "".join(buf)
            if segment:
                parts.append((segment, in_math))
            buf = []
            in_math = not in_math
            continue
        buf.append(ch)
    tail = "".join(buf)
    if tail:
        parts.append((tail, in_math))
    return parts


@dataclass(frozen=True)
class _Token:
    text: str
    is_math: bool

    def is_space(self) -> bool:
        return (not self.is_math) and self.text.isspace()


def _tokenize_plain_segment(text: str) -> list[_Token]:
    if not text:
        return []
    if _contains_cjk(text):
        return [_Token(ch, False) for ch in text]
    parts = re.findall(r"\s+|[^\s]+", text)
    return [_Token(part, False) for part in parts]


def _tokenize_mixed_line(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    for seg, is_math in _split_inline_math(text):
        if not seg:
            continue
        if is_math:
            tokens.append(_Token(seg, True))
        else:
            tokens.extend(_tokenize_plain_segment(seg))
    return tokens


def make_mixed_text_mobject(text: str, font: Optional[str], font_size: float) -> Mobject:
    # 处理含 $...$ 的行内公式，文本用 Text，公式用 MathTex
    if font is None:
        font = "Sans"
    lines = text.splitlines() if text else [text]
    line_mobjects: list[Mobject] = []
    for line in lines:
        segments = _split_inline_math(line)
        mobjects: list[Mobject] = []
        for seg, is_math in segments:
            if not seg:
                continue
            if is_math:
                # 数学段落里如果混入中文，直接用 Text，避免 MathTex 报错
                if _contains_cjk(seg):
                    mobjects.append(Text(seg, font=font, font_size=font_size) if font else Text(seg, font_size=font_size))
                    continue
                try:
                    mobjects.append(MathTex(_normalize_latex(seg), font_size=font_size))
                except Exception:
                    mobjects.append(Text(seg, font=font, font_size=font_size) if font else Text(seg, font_size=font_size))
            else:
                mobjects.append(Text(seg, font=font, font_size=font_size) if font else Text(seg, font_size=font_size))
        if not mobjects:
            line_mobjects.append(Text(line, font=font, font_size=font_size) if font else Text(line, font_size=font_size))
        elif len(mobjects) == 1:
            line_mobjects.append(mobjects[0])
        else:
            line_mobjects.append(VGroup(*mobjects).arrange(RIGHT, buff=0.05, aligned_edge=DOWN))
    if not line_mobjects:
        return Text(text, font=font, font_size=font_size) if font else Text(text, font_size=font_size)
    if len(line_mobjects) == 1:
        return line_mobjects[0]
    return VGroup(*line_mobjects).arrange(DOWN, aligned_edge=LEFT, buff=0.15)


def _build_text_mobject(text: str, font: Optional[str], font_size: float) -> Mobject:
    # 含 $...$ 的行内公式采用混排
    if "$" in text:
        return make_mixed_text_mobject(text, font=font, font_size=font_size)
    # 纯 LaTeX 语法则创建 MathTex
    if is_latex(text):
        try:
            return MathTex(_normalize_latex(text), font_size=font_size)
        except Exception:
            # LaTeX 编译失败时退化为普通文本，避免渲染中断
            if font:
                return Text(text, font=font, font_size=font_size)
            return Text(text, font_size=font_size)
    # 普通文本：指定字体则用指定字体，否则回退到 Sans
    if font is None:
        font = "Sans"
    return Text(text, font=font, font_size=font_size)


@lru_cache(maxsize=512)
def _cached_text_mobject(text: str, font: Optional[str], font_size: float) -> Mobject:
    return _build_text_mobject(text, font, font_size)


@lru_cache(maxsize=4096)
def _measure_text_width(text: str, font: Optional[str], font_size: float) -> float:
    if not text:
        return 0.0
    return _cached_text_mobject(text, font, font_size).width


@lru_cache(maxsize=4096)
def _measure_math_width(text: str, font: Optional[str], font_size: float) -> float:
    if not text:
        return 0.0
    # Force MathTex for inline math segments.
    return _measure_text_width(f"${text}$", font, font_size)


def _render_tokens(tokens: list[_Token]) -> str:
    if not tokens:
        return ""
    end = len(tokens)
    while end > 0 and tokens[end - 1].is_space():
        end -= 1
    if end <= 0:
        return ""
    out = []
    for token in tokens[:end]:
        if token.is_math:
            out.append(f"${token.text}$")
        else:
            out.append(token.text)
    return "".join(out)


def _token_width(token: _Token, font: Optional[str], font_size: float) -> float:
    if not token.text:
        return 0.0
    if token.is_math:
        return _measure_math_width(token.text, font, font_size)
    return _measure_text_width(token.text, font, font_size)


def _wrap_tokens_to_width(tokens: list[_Token], max_width: float, font: Optional[str], font_size: float) -> str:
    if not tokens:
        return ""
    if max_width <= 0:
        return _render_tokens(tokens)
    lines: list[str] = []
    buf: list[_Token] = []
    buf_width = 0.0
    for token in tokens:
        if not buf and token.is_space():
            continue
        token_width = _token_width(token, font, font_size)
        if buf and buf_width + token_width > max_width:
            line = _render_tokens(buf)
            if line or lines:
                lines.append(line)
            buf = []
            buf_width = 0.0
            if token.is_space():
                continue
        buf.append(token)
        buf_width += token_width
    if buf:
        lines.append(_render_tokens(buf))
    return "\n".join(lines)


def make_text_mobject(text: str, font: Optional[str] = None, font_size: float = 36) -> Mobject:
    """
    根据文本类型创建对应的Manim文本对象（自动区分普通文本/LaTeX公式）
    :param text: 待渲染的文本（普通文本或LaTeX公式）
    :param font: 字体名称（仅普通文本生效，None使用Manim默认字体）
    :param font_size: 文本字号（默认36）
    :return: MathTex（LaTeX公式）或Text（普通文本）对象
    """
    # 复用缓存对象，再拷贝一份避免被多处 add 造成联动
    return _cached_text_mobject(text, font, font_size).copy()


def clear_text_cache() -> None:
    _cached_text_mobject.cache_clear()
    _measure_text_width.cache_clear()
    _measure_math_width.cache_clear()


def fit_text_to_box(mobj: Mobject, max_width: float, max_height: float) -> Mobject:
    """
    将文本对象缩放到适配指定的最大宽高（优先适配宽度，再适配高度）
    :param mobj: 待缩放的Manim文本对象（Text/MathTex/VGroup等）
    :param max_width: 允许的最大宽度（防止文本超出场景）
    :param max_height: 允许的最大高度
    :return: 缩放后的文本对象
    """
    # 宽度超出则缩放到最大宽度
    if mobj.width > max_width:
        mobj.scale_to_fit_width(max_width)
    # 高度仍超出则缩放到最大高度（宽度适配后可能高度仍超）
    if mobj.height > max_height:
        mobj.scale_to_fit_height(max_height)
    return mobj


def fit_text_to_box_with_constraints(
    mobj: Mobject,
    max_width: float,
    max_height: float,
    *,
    min_font_size: Optional[float],
    text: Optional[str],
    font: Optional[str],
    font_size: Optional[float],
) -> Mobject:
    if mobj.width <= max_width and mobj.height <= max_height:
        return mobj
    if min_font_size and font_size and text:
        scale = min(1.0, max_width / mobj.width, max_height / mobj.height)
        effective_size = font_size * scale
        if effective_size < min_font_size:
            wrapped = wrap_text_to_width(text, max_width, font=font, font_size=font_size)
            mobj = make_text_mobject(wrapped, font=font, font_size=font_size)
    return fit_text_to_box(mobj, max_width=max_width, max_height=max_height)


def wrap_text_to_width(text: str, max_width: float, font: Optional[str], font_size: float) -> str:
    if not text:
        return text
    lines = text.split("\n")
    wrapped: list[str] = []
    for raw_line in lines:
        if not raw_line:
            wrapped.append("")
            continue
        if "$" in raw_line:
            tokens = _tokenize_mixed_line(raw_line)
            if not tokens:
                wrapped.append(raw_line)
                continue
            wrapped.append(_wrap_tokens_to_width(tokens, max_width, font, font_size))
            continue
        if is_latex(raw_line):
            wrapped.append(raw_line)
            continue
        tokens = _tokenize_plain_segment(raw_line)
        if not tokens:
            wrapped.append(raw_line)
            continue
        wrapped.append(_wrap_tokens_to_width(tokens, max_width, font, font_size))
    return "\n".join(wrapped)


def wrap_text_to_char_limit(text: str, max_chars: int) -> str:
    if not text or max_chars <= 0:
        return text
    out_lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            out_lines.append("")
            continue
        count = 0
        in_math = False
        buf: list[str] = []
        for ch in raw_line:
            buf.append(ch)
            if ch == "$":
                in_math = not in_math
                continue
            if in_math:
                continue
            count += 1
            if count >= max_chars:
                buf.append("\n")
                count = 0
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def pin_to_corner(mobj: Mobject, corner=LEFT + UP, buffer: float = 0.4) -> Mobject:
    """
    将Manim对象固定到场景指定角落（默认左上），并保留缓冲距离
    :param mobj: 待定位的对象（文本、图形、组合对象等）
    :param corner: 目标角落（Manim方向常量组合，如LEFT+UP=左上，RIGHT+DOWN=右下）
    :param buffer: 对象与场景边缘的缓冲距离（默认0.4）
    :return: 定位后的对象
    """
    return mobj.to_corner(corner, buff=buffer)
