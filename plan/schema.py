from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisPoints:
    """
    题目分析要点数据类：存储解题所需的核心分析信息
    """
    formulas: List[str] = field(default_factory=list)    # 解题用到的公式列表（如["f'(x)=3x²-3","点斜式y-y₀=k(x-x₀)"]）
    conditions: List[str] = field(default_factory=list)  # 题目隐含/明确条件列表（如["x∈[-2,2]","f(1)=0"]）
    strategy: List[str] = field(default_factory=list)    # 解题策略列表（如["先求导找极值点","再判断区间最值"]）


@dataclass
class StepVisual:
    """
    步骤级图形变换数据类：描述单个解题步骤对应的动画变换
    """
    action: str                                     # 变换类型：move(移动),rotate(旋转),scale(缩放),color(变色),remove(移除),opacity(透明度),follow_path(沿路径),follow_path_segment(路径区间),follow_path_rotate(沿切线转向),trajectory_trace(轨迹),ghost(残影),marker(关键点)
    target_id: str                                  # 要操作的对象ID（如"block","force_F"）
    params: Dict[str, Any] = field(default_factory=dict)  # 变换参数（如{"x": 0.5, "y": 0.3}）
    duration: float = 0.3                           # 动画持续时间（秒）


@dataclass
class Step:
    """
    单步解题步骤数据类：存储每一步的解题内容和辅助信息
    """
    line: str                              # 解题步骤核心文本（如"f'(x)=3x²-3=3(x²-1)"）
    subtitle: str                          # 步骤副标题/说明（如"步骤1：求函数的导数"）
    emphasis: Optional[Dict[str, Any]] = None  # 步骤重点标注配置（如{"position": "highlight", "text": "极值点"}）
    narration: Optional[str] = None        # 可选旁白文本（用于语音/字幕闭环）
    visual_transform: Optional[List[StepVisual]] = None  # 可选：该步骤对应的图形变换列表


@dataclass
class QuestionPlan:
    """
    单道子题规划数据类：存储某一个子问题的完整解题规划
    """
    question_text: str                     # 子题文本（如"(1) 求曲线在点(1,f(1))处的切线方程"）
    analysis: AnalysisPoints = field(default_factory=AnalysisPoints)  # 该子题的分析要点
    steps: List[Step] = field(default_factory=list)  # 该子题的解题步骤列表
    layout_overrides: Optional[Dict[str, Any]] = None  # 可选布局覆盖参数（theme/constraints）
    visual: Optional[Dict[str, Any]] = None  # 可选图形描述（用于左侧辅助图）
    model_spec: Optional[Dict[str, Any]] = None  # 可选：物理层结构化描述
    diagram_spec: Optional[Dict[str, Any]] = None  # 可选：图示层结构化描述
    motion_spec: Optional[Dict[str, Any]] = None  # 可选：运动层结构化描述（轨迹/阶段/姿态）


@dataclass
class ProblemPlan:
    """
    完整题目规划数据类：存储整道数学题的所有解题规划信息
    """
    problem_full_text: str                 # 题目完整文本（包含题干+所有子题）
    stem: str                              # 题干文本（仅已知条件部分，无具体问题）
    questions: List[QuestionPlan] = field(default_factory=list)  # 所有子题的规划列表


def _analysis_from_dict(data: Dict[str, Any]) -> AnalysisPoints:
    """
    将字典数据转换为AnalysisPoints对象（反序列化）
    :param data: 包含formulas/conditions/strategy的字典
    :return: 初始化后的AnalysisPoints对象
    """
    return AnalysisPoints(
        formulas=list(data.get("formulas", [])),  # 确保为列表类型，避免None
        conditions=list(data.get("conditions", [])),
        strategy=list(data.get("strategy", [])),
    )


def _step_visual_from_dict(data: Dict[str, Any]) -> StepVisual:
    """
    将字典数据转换为StepVisual对象（反序列化）
    :param data: 包含action/target_id/params的字典
    :return: 初始化后的StepVisual对象
    """
    return StepVisual(
        action=str(data.get("action", "")),
        target_id=str(data.get("target_id", "")),
        params=dict(data.get("params", {})) if isinstance(data.get("params"), dict) else {},
        duration=float(data.get("duration", 0.3)),
    )


def _step_from_dict(data: Dict[str, Any]) -> Step:
    """
    将字典数据转换为Step对象（反序列化）
    :param data: 包含line/subtitle/emphasis/visual_transform的字典
    :return: 初始化后的Step对象
    """
    visual_transform = None
    transforms = data.get("visual_transform")
    if isinstance(transforms, list):
        visual_transform = [_step_visual_from_dict(t) for t in transforms]
    
    return Step(
        line=str(data.get("line", "")),  # 强制转为字符串，避免非字符串类型
        subtitle=str(data.get("subtitle", "")),
        emphasis=data.get("emphasis"),  # 可选字段，None则保留默认值
        narration=data.get("narration"),
        visual_transform=visual_transform,
    )


def question_from_dict(data: Dict[str, Any]) -> QuestionPlan:
    """
    将字典数据转换为QuestionPlan对象（反序列化）
    :param data: 包含question_text/analysis/steps的字典
    :return: 初始化后的QuestionPlan对象
    """
    model_spec = data.get("model_spec")
    if isinstance(model_spec, str):
        model_spec = {"text": model_spec.strip()}
    elif not isinstance(model_spec, dict):
        model_spec = None
    diagram_spec = data.get("diagram_spec")
    if isinstance(diagram_spec, str):
        diagram_spec = {"text": diagram_spec.strip()}
    elif not isinstance(diagram_spec, dict):
        diagram_spec = None
    motion_spec = data.get("motion_spec")
    if isinstance(motion_spec, str):
        motion_spec = {"text": motion_spec.strip()}
    elif not isinstance(motion_spec, dict):
        motion_spec = None

    return QuestionPlan(
        question_text=str(data.get("question_text", "")),
        analysis=_analysis_from_dict(data.get("analysis", {})),  # 嵌套解析AnalysisPoints
        steps=[_step_from_dict(s) for s in data.get("steps", [])],  # 批量解析Step列表
        layout_overrides=data.get("layout_overrides"),
        visual=data.get("visual"),
        model_spec=model_spec,
        diagram_spec=diagram_spec,
        motion_spec=motion_spec,
    )


def problem_from_dict(data: Dict[str, Any]) -> ProblemPlan:
    """
    将字典数据转换为ProblemPlan对象（反序列化）
    :param data: 包含problem_full_text/stem/questions的字典
    :return: 初始化后的ProblemPlan对象
    """
    return ProblemPlan(
        problem_full_text=str(data.get("problem_full_text", "")),
        stem=str(data.get("stem", "")),
        questions=[question_from_dict(q) for q in data.get("questions", [])],  # 批量解析QuestionPlan列表
    )


def _analysis_to_dict(a: AnalysisPoints) -> Dict[str, Any]:
    """
    将AnalysisPoints对象转换为字典（序列化）
    :param a: AnalysisPoints对象
    :return: 包含formulas/conditions/strategy的字典
    """
    return {
        "formulas": list(a.formulas),
        "conditions": list(a.conditions),
        "strategy": list(a.strategy),
    }


def _step_visual_to_dict(sv: StepVisual) -> Dict[str, Any]:
    """
    将StepVisual对象转换为字典（序列化）
    :param sv: StepVisual对象
    :return: 包含action/target_id/params/duration的字典
    """
    return {
        "action": sv.action,
        "target_id": sv.target_id,
        "params": sv.params,
        "duration": sv.duration,
    }


def _step_to_dict(s: Step) -> Dict[str, Any]:
    """
    将Step对象转换为字典（序列化）
    :param s: Step对象
    :return: 包含line/subtitle/emphasis（可选）/visual_transform（可选）的字典
    """
    out: Dict[str, Any] = {"line": s.line, "subtitle": s.subtitle}
    # 仅当emphasis非空时才加入字典，避免冗余的null值
    if s.emphasis:
        out["emphasis"] = s.emphasis
    if s.narration:
        out["narration"] = s.narration
    if s.visual_transform:
        out["visual_transform"] = [_step_visual_to_dict(t) for t in s.visual_transform]
    return out


def question_to_dict(q: QuestionPlan) -> Dict[str, Any]:
    """
    将QuestionPlan对象转换为字典（序列化）
    :param q: QuestionPlan对象
    :return: 包含question_text/analysis/steps的字典
    """
    out = {
        "question_text": q.question_text,
        "analysis": _analysis_to_dict(q.analysis),  # 嵌套序列化AnalysisPoints
        "steps": [_step_to_dict(s) for s in q.steps],  # 批量序列化Step列表
    }
    if q.layout_overrides:
        out["layout_overrides"] = q.layout_overrides
    if q.visual:
        out["visual"] = q.visual
    if q.model_spec:
        out["model_spec"] = q.model_spec
    if q.diagram_spec:
        out["diagram_spec"] = q.diagram_spec
    if q.motion_spec:
        out["motion_spec"] = q.motion_spec
    return out


def problem_to_dict(p: ProblemPlan) -> Dict[str, Any]:
    """
    将ProblemPlan对象转换为字典（序列化）
    :param p: ProblemPlan对象
    :return: 包含problem_full_text/stem/questions的字典
    """
    return {
        "problem_full_text": p.problem_full_text,
        "stem": p.stem,
        "questions": [question_to_dict(q) for q in p.questions],  # 批量序列化QuestionPlan列表
    }
