"""原始数据源加载：统一转换为 DataFrame[source, section, question, answer, url]。

支持：
- Markdown 题库（``### 题目`` 为题干，其后正文为答案，``##`` 为章节）
- JSONL / CSV（需包含 question / answer 列，可选 section / url）
- HTML 页面（h2/h3 题干 + 段落答案，适配常见面经博客结构）
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

COLUMNS = ["source", "section", "question", "answer", "url"]

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_HEADING = re.compile(r"^(#{2,4})\s+(.*)$")


def load_markdown(path: Path) -> pd.DataFrame:
    text = _FRONT_MATTER.sub("", path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    section, question, buf = "", None, []

    def flush() -> None:
        if question is not None:
            rows.append({"section": section, "question": question, "answer": "\n".join(buf).strip()})

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m and len(m.group(1)) == 2:
            flush()
            section, question, buf = m.group(2), None, []
        elif m and len(m.group(1)) >= 3:
            flush()
            question, buf = m.group(2), []
        elif question is not None:
            buf.append(line)
    flush()

    df = pd.DataFrame(rows, columns=["section", "question", "answer"])
    df["source"] = path.stem
    df["url"] = ""
    return df[COLUMNS]


def load_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return _normalize(pd.DataFrame(records), path.stem)


def load_csv(path: Path) -> pd.DataFrame:
    return _normalize(pd.read_csv(path), path.stem)


def load_html(path: Path) -> pd.DataFrame:
    html = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"(?i)<h[23][^>]*>(.*?)</h[23]>", html)
    # split 结果：[前言, 标题1, 正文1, 标题2, 正文2, ...]
    rows = [
        {"section": "", "question": parts[i], "answer": parts[i + 1] if i + 1 < len(parts) else ""}
        for i in range(1, len(parts), 2)
    ]
    df = pd.DataFrame(rows, columns=["section", "question", "answer"])
    df["source"], df["url"] = path.stem, ""
    return df[COLUMNS]


def _normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = {"question", "answer"} - set(df.columns)
    if missing:
        raise ValueError(f"{source}: 缺少列 {missing}")
    for col in ("section", "url"):
        if col not in df.columns:
            df[col] = ""
    if "source" not in df.columns:
        df["source"] = source
    return df[COLUMNS].fillna("").astype(str)


LOADERS = {".md": load_markdown, ".jsonl": load_jsonl, ".csv": load_csv, ".html": load_html, ".htm": load_html}


def load_sources(paths: Iterable[Path]) -> pd.DataFrame:
    """递归加载目录 / 文件，返回合并后的原始题目表。"""
    frames = []
    for p in paths:
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in LOADERS) if p.is_dir() else [p]
        for f in files:
            loader = LOADERS.get(f.suffix.lower())
            if loader:
                frames.append(loader(f))
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)
