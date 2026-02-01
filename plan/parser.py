import re
from dataclasses import dataclass
from typing import List


@dataclass
class ParsedProblem:
    """
    解析后的题目数据结构：拆分题干和具体问题列表
    """
    stem: str               # 题干（题目背景/已知条件部分）
    questions: List[str]    # 具体问题列表（如(1)/(2)、1. /2.、问：等拆分后的问题）


# 问题拆分正则模式列表：匹配常见的数学题问题分隔格式，按优先级依次匹配
QUESTION_SPLIT_PATTERNS = [
    # 模式1：匹配 (1)、( 2 )、(  3  ) 等带括号的数字编号（最常见的题型分隔）
    re.compile(r"\(\s*(\d+)\s*\)"),
    # 模式2：匹配行首的 1.、2 、3、、  4. 等数字+点/顿号的编号（多行题型分隔）
    re.compile(r"^\s*(\d+)\s*[\.|、]", re.MULTILINE),
    # 模式3：匹配“问：”“问 ：”等单问题分隔（适用于只有一个问题的题型）
    re.compile(r"问\s*[:：]"),
]


def parse_stem_and_questions(problem_text: str) -> ParsedProblem:
    """
    核心解析函数：将完整题目文本拆分为题干 + 具体问题列表
    :param problem_text: 原始完整题目文本（包含题干和所有问题）
    :return: 解析后的ParsedProblem对象，包含拆分后的题干和问题列表
    """
    # 预处理：去除文本首尾空白字符
    text = problem_text.strip()
    # 边界处理：空文本直接返回空结果
    if not text:
        return ParsedProblem(stem="", questions=[])

    # 遍历拆分模式，按优先级匹配（匹配到第一个有效模式即停止）
    for pattern in QUESTION_SPLIT_PATTERNS:
        # 按正则模式拆分文本：拆分后会得到 [题干部分, 编号1, 问题1, 编号2, 问题2, ...] 结构
        parts = pattern.split(text)
        # 拆分后长度>1说明匹配到了问题分隔符
        if len(parts) > 1:
            # 提取题干（拆分后的第一个部分）并去除首尾空白
            stem = parts[0].strip()
            questions = []
            # 拆分后的剩余部分（编号+问题）
            tail = parts[1:]

            # 分支1：匹配到括号型编号（如(1)、(2)）
            if pattern.pattern.startswith("\\(") or "(" in pattern.pattern:
                # 将剩余部分转为迭代器，按“编号-问题”成对提取
                it = iter(tail)
                for _index in it:  # _index为匹配到的数字编号（如1、2），暂不使用
                    try:
                        q = next(it)  # 提取编号对应的问题文本
                    except StopIteration:  # 防止迭代器越界
                        break
                    questions.append(q.strip())  # 去除问题文本首尾空白并加入列表

            # 分支2：匹配到“问：”型分隔（单问题）
            elif "问" in pattern.pattern:
                rest = parts[1]  # 提取“问：”后的所有文本
                questions = [rest.strip()] if rest.strip() else []  # 非空则加入问题列表

            # 分支3：匹配到行首数字+点/顿号型编号（如1.、2、）
            else:
                # 同括号型逻辑，按“编号-问题”成对提取
                it = iter(tail)
                for _index in it:
                    try:
                        q = next(it)
                    except StopIteration:
                        break
                    questions.append(q.strip())

            # 过滤空问题（去除纯空白的问题文本）
            questions = [q for q in questions if q]
            # 匹配到有效问题则返回解析结果
            if questions:
                return ParsedProblem(stem=stem, questions=questions)

    # 所有模式都未匹配到：说明只有题干无分隔问题，问题列表为空
    return ParsedProblem(stem=text, questions=[])