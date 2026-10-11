"""样本质量打分：规则分（零成本、全量）+ LLM 分（可选，只对规则分处于灰区的样本调用）。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

_CODE_FENCE = re.compile(r"```")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_PLACEHOLDER = re.compile(r"(待补充|略|TODO|todo|见上文|同上|参考链接)\s*$")


@dataclass(frozen=True)
class QualityConfig:
    min_question_len: int = 6
    max_question_len: int = 200
    min_answer_len: int = 30
    max_answer_len: int = 6000
    keep_threshold: float = 0.6
    # 规则分落在 [gray_low, gray_high) 的样本交给 LLM 复核
    gray_low: float = 0.4
    gray_high: float = 0.7


def rule_score(question: str, answer: str, cfg: QualityConfig = QualityConfig()) -> tuple[float, list[str]]:
    """返回 0~1 的规则质量分与扣分原因。"""
    reasons: list[str] = []
    score = 1.0
    qlen, alen = len(question), len(answer)

    if qlen < cfg.min_question_len:
        score -= 0.5
        reasons.append("question_too_short")
    if qlen > cfg.max_question_len:
        score -= 0.2
        reasons.append("question_too_long")
    if alen < cfg.min_answer_len:
        score -= 0.5
        reasons.append("answer_too_short")
    if alen > cfg.max_answer_len:
        score -= 0.1
        reasons.append("answer_too_long")
    if _PLACEHOLDER.search(answer):
        score -= 0.4
        reasons.append("answer_placeholder")
    if answer and len(_CODE_FENCE.findall(answer)) % 2 == 1:
        score -= 0.1
        reasons.append("unclosed_code_block")
    if answer and len(_CJK.findall(answer)) / max(alen, 1) < 0.05 and "```" not in answer:
        score -= 0.2
        reasons.append("low_cjk_ratio")
    if question and answer and answer.strip().startswith(question.strip()[:20]) and alen < qlen * 2:
        score -= 0.3
        reasons.append("answer_repeats_question")
    return max(0.0, round(score, 3)), reasons


LLM_JUDGE_SYSTEM = "你是面试题库质量审核员，判断一条面试题及参考答案是否适合进入高质量题库。只输出 JSON。"
LLM_JUDGE_USER = """评估维度：题目是否清晰具体、答案是否正确完整、是否有实际面试价值。
输出 JSON：{{"score": 0-10 的整数, "reason": "一句话理由"}}

题目：{question}
参考答案：{answer}"""


def score_frame(
    df: pd.DataFrame,
    cfg: QualityConfig = QualityConfig(),
    llm_judge: Callable[[str, str], float] | None = None,
) -> pd.DataFrame:
    """新增 rule_score / quality_reasons / llm_score / quality_score / keep 列。

    quality_score：有 LLM 分时取 0.4*规则分 + 0.6*LLM 分，否则取规则分。
    """
    out = df.copy()
    scored = [rule_score(q, a, cfg) for q, a in zip(out["question"], out["answer"])]
    out["rule_score"] = [s for s, _ in scored]
    out["quality_reasons"] = [",".join(r) for _, r in scored]
    out["llm_score"] = float("nan")

    if llm_judge is not None:
        gray = out["rule_score"].between(cfg.gray_low, cfg.gray_high, inclusive="left")
        for idx in out.index[gray]:
            try:
                out.at[idx, "llm_score"] = llm_judge(out.at[idx, "question"], out.at[idx, "answer"])
            except Exception:  # noqa: BLE001 - 单条失败不影响整体，保留规则分
                continue

    has_llm = out["llm_score"].notna()
    out["quality_score"] = out["rule_score"]
    out.loc[has_llm, "quality_score"] = 0.4 * out.loc[has_llm, "rule_score"] + 0.6 * out.loc[has_llm, "llm_score"]
    out["keep"] = out["quality_score"] >= cfg.keep_threshold
    return out


def make_llm_judge(client) -> Callable[[str, str], float]:
    def judge(question: str, answer: str) -> float:
        data = client.chat_json(LLM_JUDGE_SYSTEM, LLM_JUDGE_USER.format(question=question, answer=answer[:3000]))
        return max(0.0, min(10.0, float(data.get("score", 0)))) / 10.0

    return judge
