"""把历史面试问答整理成评分模型的 SFT 数据集。

数据来源（二选一）：
- PostgreSQL interview_records.report_json.detailed_review（线上真实问答 + LLM 评分）
- JSONL 导出文件，每行至少包含 question / answer / score，可选 reference / feedback / key_points_hit / key_points_missed

标签对齐：
- 若提供人工标注文件（question+answer → human_score），人工分优先覆盖模型分（label_source=human）
- 分数统一裁剪到 [0, 100] 并四舍五入，派生 5 档 score_bucket 用于分层划分与一致性评估

输出：train.jsonl / val.jsonl（chat messages 格式，可直接用于 Transformers chat template）、
manual_review.csv（分层抽样的人工抽检表）、stats.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..prompts import build_scorer_messages, format_scorer_target

log = logging.getLogger("interview_ml.sft")

BUCKETS = [0, 30, 50, 70, 90, 101]
BUCKET_LABELS = ["0-29", "30-49", "50-69", "70-89", "90-100"]


def load_from_postgres(dsn: str, limit: int | None = None) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    sql = """
        SELECT r.session_id, r.position, d->>'question_content' AS question, d->>'user_answer' AS answer,
               (d->>'score')::float8 AS score, d->>'comment' AS feedback,
               COALESCE(d->'key_points_hit', '[]'::jsonb)::text AS key_points_hit,
               COALESCE(d->'key_points_missed', '[]'::jsonb)::text AS key_points_missed
        FROM interview_records r, LATERAL jsonb_array_elements(r.report_json->'detailed_review') d
        ORDER BY r.created_at
    """ + (f" LIMIT {int(limit)}" if limit else "")
    with create_engine(dsn).connect() as conn:
        df = pd.read_sql(text(sql), conn)
    for col in ("key_points_hit", "key_points_missed"):
        df[col] = df[col].map(json.loads)
    df["reference"] = ""
    return df


def load_from_jsonl(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    for col, default in (("reference", ""), ("feedback", ""), ("key_points_hit", None), ("key_points_missed", None)):
        if col not in df.columns:
            df[col] = default
    for col in ("key_points_hit", "key_points_missed"):
        df[col] = df[col].map(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [])
    return df


def _pair_key(question: str, answer: str) -> str:
    return hashlib.sha1(f"{question.strip()}\n{answer.strip()}".encode()).hexdigest()


def align_labels(df: pd.DataFrame, human: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    out["question"] = out["question"].fillna("").astype(str).str.strip()
    out["answer"] = out["answer"].fillna("").astype(str).str.strip()
    out = out[(out["question"].str.len() > 0)].copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["label_source"] = "model"
    out["pair_key"] = [_pair_key(q, a) for q, a in zip(out["question"], out["answer"])]

    if human is not None and not human.empty:
        h = human.assign(pair_key=[_pair_key(q, a) for q, a in zip(human["question"], human["answer"])])
        h = h.drop_duplicates("pair_key", keep="last").set_index("pair_key")["human_score"]
        mask = out["pair_key"].isin(h.index)
        out.loc[mask, "score"] = out.loc[mask, "pair_key"].map(h)
        out.loc[mask, "label_source"] = "human"

    out = out.dropna(subset=["score"])
    out["score"] = out["score"].clip(0, 100).round().astype(int)
    # 空回答 / "不会" 类样本强制低分，修正上游评分偶发的不一致
    blank = out["answer"].str.fullmatch(r"(不会|不知道|不清楚|跳过|pass|)?", case=False)
    out.loc[blank, "score"] = out.loc[blank, "score"].clip(upper=10)
    out["score_bucket"] = pd.cut(out["score"], BUCKETS, right=False, labels=BUCKET_LABELS).astype(str)
    return out.drop_duplicates("pair_key", keep="last").reset_index(drop=True)


def to_messages(row: pd.Series) -> dict:
    messages = build_scorer_messages(row["question"], row["answer"], row.get("reference") or "")
    messages.append(
        {
            "role": "assistant",
            "content": format_scorer_target(
                row["score"], row.get("feedback") or "", list(row["key_points_hit"]), list(row["key_points_missed"])
            ),
        }
    )
    return {"messages": messages, "score": int(row["score"]), "label_source": row["label_source"]}


def stratified_split(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 score_bucket 分层划分；同一题目只出现在一侧，避免题目泄漏导致验证集虚高。"""
    rng = np.random.default_rng(seed)
    val_questions: set[str] = set()
    for _, group in df.groupby("score_bucket"):
        questions = group["question"].unique()
        n_val = max(1, int(round(len(questions) * val_ratio))) if len(questions) > 1 else 0
        val_questions.update(rng.choice(questions, size=n_val, replace=False).tolist())
    is_val = df["question"].isin(val_questions)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def manual_review_sample(df: pd.DataFrame, per_bucket: int = 20, seed: int = 42) -> pd.DataFrame:
    sample = df.groupby("score_bucket", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_bucket), random_state=seed)
    )
    return sample[["pair_key", "question", "answer", "score", "score_bucket", "label_source"]].assign(
        reviewer_score="", reviewer_note=""
    )


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="构建评分模型 SFT 数据集")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-postgres", action="store_true")
    src.add_argument("--from-jsonl", type=Path)
    ap.add_argument("--human-labels", type=Path, help="人工标注 CSV/JSONL：question, answer, human_score")
    ap.add_argument("--out", type=Path, default=Path("build/sft"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.from_postgres:
        from ..config import get_settings

        raw = load_from_postgres(get_settings().pg_dsn)
    else:
        raw = load_from_jsonl(args.from_jsonl)

    human = None
    if args.human_labels:
        human = pd.read_csv(args.human_labels) if args.human_labels.suffix == ".csv" else pd.read_json(
            args.human_labels, lines=True
        )

    data = align_labels(raw, human)
    train, val = stratified_split(data, args.val_ratio, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl([to_messages(r) for _, r in train.iterrows()], args.out / "train.jsonl")
    write_jsonl([to_messages(r) for _, r in val.iterrows()], args.out / "val.jsonl")
    manual_review_sample(data, seed=args.seed).to_csv(args.out / "manual_review.csv", index=False)
    stats = {
        "raw": len(raw),
        "aligned": len(data),
        "train": len(train),
        "val": len(val),
        "human_labeled": int((data["label_source"] == "human").sum()),
        "bucket_distribution": data["score_bucket"].value_counts().to_dict(),
    }
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("SFT 数据集：%s", stats)


if __name__ == "__main__":
    main()
