from __future__ import annotations

from typing import Iterable, Optional

# Manim核心组件：用于创建图形、文本、组合对象及方向常量
from manim import DOWN, LEFT, RIGHT, UP, VGroup, Rectangle, Text, Mobject

# 自定义文本工具：fit_text_to_box（文本适配指定尺寸）、make_text_mobject（创建文本对象）
from .text_fit import fit_text_to_box, fit_text_to_box_with_constraints, make_text_mobject, wrap_text_to_width
from .theme import Constraints, Theme


class PinnedHeader(VGroup):
    """
    固定显示的题目头部组件（带边框）
    用于教学视频中固定展示题干+问题文本，自动适配指定尺寸并添加透明边框
    """
    def __init__(
        self,
        stem: str,               # 题干文本（如“已知函数f(x)=x³-3x+a”）
        question_text: str,      # 问题文本（如“求切线方程”）
        max_width: float,        # 组件最大宽度（防止超出场景）
        max_height: float,       # 组件最大高度
        font: Optional[str] = None,  # 字体（None使用Manim默认字体）
        font_size: float = 28,   # 文本字号
        theme: Optional[Theme] = None,
        constraints: Optional[Constraints] = None,
    ) -> None:
        super().__init__()
        theme = theme or Theme()
        constraints = constraints or Constraints()
        font = font or theme.font
        font_size = font_size * theme.font_size_scale
        # 1. 创建题干和问题的文本对象
        stem_text = wrap_text_to_width(stem, max_width, font=font, font_size=font_size)
        q_text = wrap_text_to_width(question_text, max_width, font=font, font_size=font_size)
        stem_m = make_text_mobject(stem_text, font=font, font_size=font_size)
        q_m = make_text_mobject(q_text, font=font, font_size=font_size)
        # 2. 垂直组合文本（左对齐，上下间距跟随行距比例）
        content = VGroup(stem_m, q_m).arrange(DOWN, aligned_edge=LEFT, buff=0.2 * theme.line_spacing)
        # 3. 适配文本到指定最大尺寸（防止溢出）
        fit_text_to_box(content, max_width=max_width, max_height=max_height)
        # 4. 创建透明边框（padding由主题控制）
        pad = theme.panel_padding
        frame = Rectangle(width=content.width + pad * 2, height=content.height + pad * 2)
        frame.set_stroke(width=2)    # 设置边框描边宽度
        frame.set_fill(opacity=0)    # 边框填充透明
        # 5. 将边框和文本添加到组件中
        self.add(frame, content)
        # 暴露子组件供外部调整（如位置、样式）
        self.content = content
        self.frame = frame


class AnalysisPanel(VGroup):
    """
    分析面板组件（带标题、有序列表、边框）
    用于展示解题分析要点，自动生成有序列表并适配尺寸，添加透明边框
    """
    def __init__(
        self,
        title: str,              # 面板标题（如“解题思路”）
        items: Iterable[str],    # 分析要点列表（如["求导","找极值点"]）
        max_width: float,        # 面板最大宽度
        max_height: float,       # 面板最大高度
        font: Optional[str] = None,  # 字体
        font_size: float = 28,   # 文本字号
        theme: Optional[Theme] = None,
        constraints: Optional[Constraints] = None,
    ) -> None:
        super().__init__()
        theme = theme or Theme()
        constraints = constraints or Constraints()
        font = font or theme.font
        font_size = font_size * theme.font_size_scale
        # 1. 创建面板标题文本
        title_m = Text(title, font=font, font_size=font_size) if font else Text(title, font_size=font_size)
        # 2. 生成有序列表项（1. xxx、2. xxx...）
        lines = []
        for i, item in enumerate(items, start=1):
            # 编号始终用普通文本，内容交给 make_text_mobject 自动处理
            idx = Text(f"{i}.", font=font, font_size=font_size) if font else Text(f"{i}.", font_size=font_size)
            wrapped = wrap_text_to_width(item, max_width, font=font, font_size=font_size)
            content = make_text_mobject(wrapped, font=font, font_size=font_size)
            line = VGroup(idx, content).arrange(RIGHT, buff=0.15, aligned_edge=DOWN)
            lines.append(line)
        # 3. 垂直组合列表项（左对齐，上下间距跟随行距比例），无内容则为空VGroup
        body = VGroup(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.15 * theme.line_spacing) if lines else VGroup()
        # 4. 组合标题和列表项（左对齐，上下间距跟随行距比例）
        panel = VGroup(title_m, body).arrange(DOWN, aligned_edge=LEFT, buff=0.2 * theme.line_spacing)
        # 5. 适配面板到指定最大尺寸
        fit_text_to_box(panel, max_width=max_width, max_height=max_height)
        # 6. 创建透明边框（padding由主题控制）
        pad = theme.panel_padding
        box = Rectangle(width=panel.width + pad * 2, height=panel.height + pad * 2)
        box.set_stroke(width=2)
        box.set_fill(opacity=0)
        # 7. 添加边框和面板内容到组件
        self.add(box, panel)
        # 暴露子组件供外部调整
        self.panel = panel
        self.box = box


class SolutionLine(VGroup):
    """
    解题步骤单行文本组件
    用于展示单行解题公式/步骤，自动适配指定宽度，固定最大高度为2
    """
    def __init__(
        self,
        line: str,               # 单行解题文本（如“f'(x)=3x²-3”）
        max_width: float,        # 最大宽度（防止超出场景）
        font: Optional[str] = None,  # 字体
        font_size: float = 34,   # 文本字号（解题步骤字号更大）
        theme: Optional[Theme] = None,
        constraints: Optional[Constraints] = None,
    ) -> None:
        super().__init__()
        theme = theme or Theme()
        constraints = constraints or Constraints()
        font = font or theme.font
        font_size = font_size * theme.font_size_scale
        # 1. 创建单行文本对象
        wrapped = wrap_text_to_width(line, max_width, font=font, font_size=font_size)
        text_m = make_text_mobject(wrapped, font=font, font_size=font_size)
        # 2. 适配文本到指定宽度（最大高度固定为2）
        fit_text_to_box_with_constraints(
            text_m,
            max_width=max_width,
            max_height=2,
            min_font_size=constraints.min_font_size,
            text=wrapped,
            font=font,
            font_size=font_size,
        )
        # 3. 添加文本到组件
        self.add(text_m)
        # 暴露文本对象供外部调整
        self.text_m = text_m


class Subtitle(VGroup):
    """
    副标题文本组件
    用于展示视频/步骤的副标题，自动适配指定宽度，固定最大高度为1
    """
    def __init__(
        self,
        text: str,               # 副标题文本（如“步骤1：求函数的导数”）
        max_width: float,        # 最大宽度
        font: Optional[str] = None,  # 字体
        font_size: float = 28,   # 文本字号
        theme: Optional[Theme] = None,
        constraints: Optional[Constraints] = None,
    ) -> None:
        super().__init__()
        theme = theme or Theme()
        constraints = constraints or Constraints()
        font = font or theme.font
        font_size = font_size * theme.font_size_scale
        # 1. 创建副标题文本对象（支持 $...$ 行内公式混排）
        wrapped = wrap_text_to_width(text, max_width, font=font, font_size=font_size)
        text_m = make_text_mobject(wrapped, font=font, font_size=font_size)
        # 2. 适配文本到指定宽度（最大高度固定为1）
        fit_text_to_box_with_constraints(
            text_m,
            max_width=max_width,
            max_height=1,
            min_font_size=constraints.min_font_size,
            text=wrapped,
            font=font,
            font_size=font_size,
        )
        # 3. 添加可选背景
        if theme.subtitle_bg_opacity > 0:
            pad = theme.panel_padding
            box = Rectangle(width=text_m.width + pad * 2, height=text_m.height + pad)
            box.set_fill(color=theme.accent_color, opacity=theme.subtitle_bg_opacity)
            box.set_stroke(opacity=0)
            self.add(box, text_m)
            self.box = box
        else:
            self.add(text_m)
        # 暴露文本对象供外部调整
        self.text_m = text_m


def make_full_problem(text: str, max_width: float, max_height: float, font: Optional[str] = None) -> Mobject:
    """
    创建完整题目文本组件（大字号）
    用于展示完整的题目内容，自动适配指定尺寸，字号固定为40（突出题目）
    :param text: 完整题目文本
    :param max_width: 最大宽度
    :param max_height: 最大高度
    :param font: 字体
    :return: 适配后的题目文本Mobject
    """
    # 创建大字号题目文本（字号40）
    mobj = make_text_mobject(text, font=font, font_size=40)
    # 适配文本到指定尺寸并返回
    return fit_text_to_box(mobj, max_width=max_width, max_height=max_height)
