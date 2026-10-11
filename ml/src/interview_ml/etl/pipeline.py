"""题库 ETL 流水线 CLI。

示例：
    iml-etl --input ../data/questions --owner alice --out build/etl            # 仅规则，产出 parquet/jsonl + 报告
    iml-etl --input raw/ --owner alice --llm-judge --llm-extract --to-postgres --to-milvus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from . import cleaning, extract, minhash, quality, sources

log = logging.getLogger("interview_ml.etl")


@dataclass
class PipelineReport:
    stages: dict[str, int] = field(default_factory=dict)
    drop_reasons: dict[str, int] = field(default_factory=dict)
    difficulty: dict[str, int] = field(default_factory=dict)
    topics: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def run(
    inputs: list[Path],
    owner: str,
    dedup_threshold: float = 0.8,
    quality_cfg: quality.QualityConfig = quality.QualityConfig(),
    llm_judge=None,
    llm_extractor=None,
) -> tuple[pd.DataFrame, PipelineReport]:
    t0 = time.time()
    report = PipelineReport()

    raw = sources.load_sources(inputs)
    report.stages["raw"] = len(raw)

    cleaned = cleaning.clean_frame(raw)
    report.stages["cleaned"] = len(cleaned)

    # 精确去重（规范化题干）→ 近似去重（MinHash LSH）
    cleaned["content_hash"] = cleaned["question"].map(
        lambda q: hashlib.sha256(minhash._NON_WORD.sub("", q.lower()).encode()).hexdigest()
    )
    exact = cleaned.drop_duplicates("content_hash").reset_index(drop=True)
    report.stages["exact_dedup"] = len(exact)
    deduped = minhash.dedup_frame(exact, text_col="question", threshold=dedup_threshold)
    report.stages["near_dedup"] = len(deduped)

    scored = quality.score_frame(deduped, quality_cfg, llm_judge=llm_judge)
    for reasons in scored.loc[~scored["keep"], "quality_reasons"]:
        for r in filter(None, reasons.split(",")):
            report.drop_reasons[r] = report.drop_reasons.get(r, 0) + 1
    kept = scored[scored["keep"]].reset_index(drop=True)
    report.stages["quality_kept"] = len(kept)

    structured = extract.extract_frame(kept, owner=owner, llm_extractor=llm_extractor)
    if not structured.empty:
        structured["content_hash"] = kept["content_hash"].values
        report.difficulty = structured["difficulty"].value_counts().to_dict()
        report.topics = structured["topic"].value_counts().head(20).to_dict()
    report.stages["final"] = len(structured)
    report.elapsed_seconds = round(time.time() - t0, 2)
    return structured, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="InterviewAgent 题库 ETL")
    ap.add_argument("--input", nargs="+", type=Path, required=True, help="题库文件或目录（md/jsonl/csv/html）")
    ap.add_argument("--owner", default="public", help="题库归属用户（与后端登录用户名一致，用于用户级隔离）")
    ap.add_argument("--out", type=Path, default=Path("build/etl"))
    ap.add_argument("--dedup-threshold", type=float, default=0.8)
    ap.add_argument("--keep-threshold", type=float, default=0.6)
    ap.add_argument("--llm-judge", action="store_true", help="对规则灰区样本调用 LLM 打分")
    ap.add_argument("--llm-extract", action="store_true", help="用 LLM 抽取知识点 / 难度")
    ap.add_argument("--to-postgres", action="store_true")
    ap.add_argument("--to-milvus", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = None
    if args.llm_judge or args.llm_extract or args.to_milvus:
        from ..llm import LLMClient

        client = LLMClient()

    df, report = run(
        args.input,
        owner=args.owner,
        dedup_threshold=args.dedup_threshold,
        quality_cfg=quality.QualityConfig(keep_threshold=args.keep_threshold),
        llm_judge=quality.make_llm_judge(client) if args.llm_judge else None,
        llm_extractor=extract.make_llm_extractor(client) if args.llm_extract else None,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_json(args.out / "questions.jsonl", orient="records", lines=True, force_ascii=False)
    (args.out / "report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("ETL 完成：%s", json.dumps(report.stages, ensure_ascii=False))

    if args.to_postgres:
        from ..config import get_settings
        from .sinks import write_postgres

        log.info("写入 PostgreSQL：%d 条", write_postgres(df, get_settings().pg_dsn))
    if args.to_milvus:
        from ..config import get_settings
        from .sinks import write_milvus

        s = get_settings()
        log.info("写入 Milvus：%d 条", write_milvus(df, s.milvus_uri, s.milvus_collection, client.embed))


if __name__ == "__main__":
    main()
