"""评分 Prompt 模板。SFT 数据构建、训练、在线推理共用同一份模板，保证训练 / 推理分布一致。"""

from __future__ import annotations

import json
from typing import Any

SCORER_SYSTEM = "你是一个严格客观的技术面试评分专家。只根据候选人明确表述的内容评分，输出纯 JSON。"

SCORER_USER_TEMPLATE = """请对候选人的回答进行评分。

题目：{question}
候选人回答：{answer}
参考答案要点：{reference}

输出 JSON：{{"score": 0-100 的整数, "feedback": "具体反馈", "key_points_hit": [...], "key_points_missed": [...], "should_follow_up": true/false}}
评分标准：90-100 完美；70-89 良好；50-69 基本，有明显遗漏；30-49 较差；0-29 未能回答或偏题。"""


def build_scorer_messages(question: str, answer: str, reference: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SCORER_SYSTEM},
        {
            "role": "user",
            "content": SCORER_USER_TEMPLATE.format(
                question=question.strip(), answer=answer.strip() or "（未作答）", reference=reference.strip() or "无参考答案"
            ),
        },
    ]


def format_scorer_target(
    score: int,
    feedback: str,
    key_points_hit: list[str],
    key_points_missed: list[str],
) -> str:
    score = int(score)
    target: dict[str, Any] = {
        "score": score,
        "feedback": feedback,
        "key_points_hit": key_points_hit,
        "key_points_missed": key_points_missed,
        "should_follow_up": 30 <= score < 80 and bool(key_points_missed),
    }
    return json.dumps(target, ensure_ascii=False)


RESUME_EXTRACT_SYSTEM = "你是简历结构化信息抽取助手。只抽取文本中明确出现的信息，不要编造，输出纯 JSON。"

RESUME_EXTRACT_USER = """从下面的简历片段中抽取结构化信息，片段可能不完整，没有的字段输出空数组 / 空字符串。

输出 JSON：
{{
  "name": "",
  "skills": ["技术栈 / 工具 / 框架，统一为常见写法，如 Go、MySQL、Redis、Kafka"],
  "education": [{{"school": "", "degree": "", "major": "", "year": ""}}],
  "experience": [{{"company": "", "title": "", "duration": "", "description": "", "tech_stack": []}}],
  "projects": [{{"name": "", "role": "", "description": "", "tech_stack": [], "highlights": ["量化结果 / 技术亮点"]}}]
}}

简历片段：
{chunk}"""
