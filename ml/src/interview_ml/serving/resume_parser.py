"""简历解析：LlamaIndex 读取 PDF / Word / 文本并按句切分成块，逐块 Prompt 抽取后合并为结构化简历。"""

from __future__ import annotations

import base64
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..prompts import RESUME_EXTRACT_SYSTEM, RESUME_EXTRACT_USER

log = logging.getLogger("interview_ml.resume")

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

ExtractFn = Callable[[str], dict[str, Any]]


def load_documents(*, text: str | None = None, filename: str | None = None, data_b64: str | None = None) -> list:
    """返回 LlamaIndex Document 列表。文件走 SimpleDirectoryReader（PDF→pypdf，DOCX→docx2txt）。"""
    from llama_index.core import Document, SimpleDirectoryReader

    if text:
        return [Document(text=text)]
    if not filename or not data_b64:
        raise ValueError("需要提供 text，或 filename + data")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型: {suffix}")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"resume{suffix}"
        path.write_bytes(base64.b64decode(data_b64))
        return SimpleDirectoryReader(input_files=[str(path)]).load_data()


def chunk_documents(docs: list, chunk_size: int = 1024, chunk_overlap: int = 128) -> list[str]:
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [n.get_content() for n in splitter.get_nodes_from_documents(docs) if n.get_content().strip()]


def _dedup(items: list, key: Callable[[Any], str]) -> list:
    seen: dict[str, Any] = {}
    for it in items:
        k = key(it).strip().lower()
        if k and k not in seen:
            seen[k] = it
        elif k in seen and isinstance(it, dict):
            # 同名项目 / 经历分在不同块时合并字段
            for f, v in it.items():
                if isinstance(v, list):
                    seen[k][f] = list(dict.fromkeys([*(seen[k].get(f) or []), *v]))
                elif v and not seen[k].get(f):
                    seen[k][f] = v
    return list(seen.values())


def merge_extractions(parts: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {"name": "", "skills": [], "education": [], "experience": [], "projects": []}
    for p in parts:
        merged["name"] = merged["name"] or (p.get("name") or "")
        merged["skills"].extend(str(s) for s in p.get("skills") or [])
        for f in ("education", "experience", "projects"):
            merged[f].extend(x for x in p.get(f) or [] if isinstance(x, dict))
    merged["skills"] = _dedup(merged["skills"], key=str)
    merged["education"] = _dedup(merged["education"], key=lambda e: f"{e.get('school', '')}{e.get('degree', '')}")
    merged["experience"] = _dedup(merged["experience"], key=lambda e: f"{e.get('company', '')}{e.get('title', '')}")
    merged["projects"] = _dedup(merged["projects"], key=lambda p: p.get("name", ""))
    return merged


class ResumeParser:
    def __init__(self, extract_fn: ExtractFn, chunk_size: int = 1024):
        self.extract_fn = extract_fn
        self.chunk_size = chunk_size

    @classmethod
    def with_llm(cls, client) -> ResumeParser:
        return cls(lambda chunk: client.chat_json(RESUME_EXTRACT_SYSTEM, RESUME_EXTRACT_USER.format(chunk=chunk)))

    def parse(self, **source: str | None) -> dict[str, Any]:
        docs = load_documents(**source)
        raw_text = "\n".join(d.get_content() for d in docs)
        chunks = chunk_documents(docs, chunk_size=self.chunk_size)
        parts = []
        for i, chunk in enumerate(chunks):
            try:
                parts.append(self.extract_fn(chunk))
            except Exception as e:  # noqa: BLE001 - 单块失败不影响其余块
                log.warning("简历块 %d 抽取失败: %s", i, e)
        result = merge_extractions(parts)
        result["raw_text"] = raw_text
        result["chunks"] = len(chunks)
        return result
