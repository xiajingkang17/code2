import json
from pathlib import Path
from typing import Any, Dict

# 导入问题规划相关的模型和转换函数：
# ProblemPlan：题目规划数据模型（核心业务对象）
# problem_from_dict：字典转ProblemPlan对象（反序列化）
# problem_to_dict：ProblemPlan对象转字典（序列化）
from .schema import ProblemPlan, problem_from_dict, problem_to_dict


def load_plan(path: str | Path) -> ProblemPlan:
    """
    从指定路径加载JSON文件，并转换为ProblemPlan对象（反序列化）
    :param path: JSON文件路径（支持字符串或Path对象）
    :return: 解析后的ProblemPlan业务对象（包含题目规划的所有信息）
    """
    # 1. 读取文件文本（UTF-8编码，兼容中文）并解析为JSON字典
    # 兼容带 BOM 的 UTF-8 文件（Windows 环境常见）
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # 2. 将JSON字典转换为ProblemPlan对象并返回
    return problem_from_dict(data)


def dump_plan(plan: ProblemPlan, path: str | Path) -> None:
    """
    将ProblemPlan对象序列化为JSON字典，并保存到指定路径（序列化）
    :param plan: 待保存的ProblemPlan业务对象
    :param path: 保存JSON文件的路径（支持字符串或Path对象）
    :return: None
    """
    # 1. 将ProblemPlan对象转换为可序列化的字典
    payload: Dict[str, Any] = problem_to_dict(plan)
    # 2. 序列化为JSON字符串（ensure_ascii=False保留中文，indent=2格式化便于阅读）
    # 3. 写入文件（UTF-8编码，确保中文正常存储）
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
