"""结果落库：PostgreSQL question_bank（与后端共用 schema）+ Milvus interview_questions（与后端 MilvusStore 同构）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_SCHEMA = Path(__file__).resolve().parents[4] / "src" / "main" / "resources" / "db" / "schema.sql"

_UPSERT = text(
    """
    INSERT INTO question_bank (id, owner_id, source_file, topic, question, reference_answer,
                               question_type, difficulty, knowledge_points, quality_score, content_hash)
    VALUES (:id, :owner_id, :source_file, :topic, :question, :reference_answer,
            :question_type, :difficulty, CAST(:knowledge_points AS jsonb), :quality_score, :content_hash)
    ON CONFLICT (id) DO UPDATE SET
        source_file = EXCLUDED.source_file, topic = EXCLUDED.topic, question = EXCLUDED.question,
        reference_answer = EXCLUDED.reference_answer, question_type = EXCLUDED.question_type,
        difficulty = EXCLUDED.difficulty, knowledge_points = EXCLUDED.knowledge_points,
        quality_score = EXCLUDED.quality_score, content_hash = EXCLUDED.content_hash, updated_at = now()
    """
)


def milvus_content(question: str, reference: str) -> str:
    """与后端 MilvusStore / QuestionBankStore 一致的检索文本格式。"""
    return f"{question}\n参考答案：{reference}"


def write_postgres(df: pd.DataFrame, dsn: str, schema_path: Path = DEFAULT_SCHEMA) -> int:
    engine = create_engine(dsn)
    with engine.begin() as conn:
        if schema_path.exists():
            for stmt in _split_sql(schema_path.read_text(encoding="utf-8")):
                conn.exec_driver_sql(stmt)
        rows = [
            {
                **{k: r[k] for k in ("id", "owner_id", "source_file", "topic", "question", "reference_answer",
                                     "question_type", "difficulty", "quality_score")},
                "knowledge_points": json.dumps(list(r["knowledge_points"]), ensure_ascii=False),
                "content_hash": r.get("content_hash"),
            }
            for r in df.to_dict("records")
        ]
        if rows:
            conn.execute(_UPSERT, rows)
    return len(rows)


def _split_sql(sql: str) -> list[str]:
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def write_milvus(df: pd.DataFrame, uri: str, collection: str, embed, batch_size: int = 10) -> int:
    from pymilvus import MilvusClient  # 可选依赖：pip install interview-ml[milvus]

    client = MilvusClient(uri=uri)
    if not client.has_collection(collection):
        raise RuntimeError(f"Milvus 集合 {collection} 不存在，请先启动一次后端完成建表")
    total = 0
    records = df.to_dict("records")
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        contents = [milvus_content(r["question"], r["reference_answer"])[:8000] for r in batch]
        vectors = embed(contents)
        client.upsert(
            collection_name=collection,
            data=[
                {
                    "id": r["id"],
                    "content": c,
                    "vector": v,
                    "user_id": r["owner_id"],
                    "source_file": r["source_file"],
                    "metadata": {
                        "type": r["question_type"],
                        "difficulty": r["difficulty"],
                        "skills": list(r["knowledge_points"]),
                        "reference": r["reference_answer"][:4000],
                        "user_id": r["owner_id"],
                        "source_file": r["source_file"],
                    },
                }
                for r, c, v in zip(batch, contents, vectors)
            ],
        )
        total += len(batch)
    return total
