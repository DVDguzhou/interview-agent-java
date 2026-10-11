"""评分模型一致性评估：对比「模型分 vs 人工标注分」。

输入为人工标注集（JSONL：question, answer, reference, human_score），对每个评分器跑一遍并输出指标：
    iml-eval --labels build/sft/human_eval.jsonl --scorer llm --scorer finetuned --out build/eval
- llm：通用大模型（DashScope，Prompt 与线上一致）作为基线
- finetuned：本地加载 base model + LoRA adapter
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ..llm import extract_json
from ..prompts import build_scorer_messages
from .metrics import agreement_report

log = logging.getLogger("interview_ml.eval")


def evaluate_scorer(labels: pd.DataFrame, score_fn: Callable[[str, str, str], float]) -> tuple[dict, list[float]]:
    preds: list[float] = []
    for row in labels.itertuples():
        try:
            preds.append(float(score_fn(row.question, row.answer, getattr(row, "reference", "") or "")))
        except Exception as e:  # noqa: BLE001
            log.warning("评分失败（记为 50）：%s", e)
            preds.append(50.0)
    return agreement_report(preds, labels["human_score"].astype(float).tolist()), preds


def llm_score_fn() -> Callable[[str, str, str], float]:
    from ..llm import LLMClient

    client = LLMClient()

    def fn(q: str, a: str, r: str) -> float:
        msgs = build_scorer_messages(q, a, r)
        return float(extract_json(client.chat(msgs[0]["content"], msgs[1]["content"]))["score"])

    return fn


def finetuned_score_fn() -> Callable[[str, str, str], float]:
    from ..serving.scorer import FineTunedScorer

    scorer = FineTunedScorer.from_settings()
    return lambda q, a, r: scorer.score(q, a, r)["score"]


SCORERS = {"llm": llm_score_fn, "finetuned": finetuned_score_fn}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="评分模型与人工标注一致性评估")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--scorer", action="append", choices=sorted(SCORERS), required=True)
    ap.add_argument("--out", type=Path, default=Path("build/eval"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    labels = pd.read_json(args.labels, lines=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.scorer:
        report, preds = evaluate_scorer(labels, SCORERS[name]())
        summary[name] = report
        labels.assign(pred=preds).to_json(args.out / f"pred_{name}.jsonl", orient="records", lines=True, force_ascii=False)
        log.info("[%s] %s", name, report)
    (args.out / "agreement.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
