"""文本清洗：HTML / Markdown 噪声去除、推广信息过滤、全半角与空白统一、题号剥离。"""

from __future__ import annotations

import html
import re
import unicodedata

import pandas as pd

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)[^)]*\)")
_ZHIDA_LINK = re.compile(r"\[`?([^\]`]+)`?\]\(https?://zhida\.zhihu\.com[^)]*\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ESCAPES = re.compile(r"\\([_\[\]*`#])")
_QUESTION_NO = re.compile(r"^\s*(?:Q?\d+(?:\.\d+)*[.、:：)]?|[（(]\d+[)）])\s*")
_MULTI_BLANK = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\u00a0\u3000]+")

# 推广 / 引流类噪声行（公众号、加群、领取资料等）
_PROMO = re.compile(r"(关注.{0,10}公众号|回复[:：]?.{0,6}领取|扫码|加群|加微信|点赞.{0,4}收藏|转载请注明|版权声明)")


_FULLWIDTH_ALNUM = {c: c - 0xFEE0 for r in ((0xFF10, 0xFF19), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)) for c in range(r[0], r[1] + 1)}


def normalize_unicode(text: str) -> str:
    # 全角字母数字转半角；中文标点（，。？）保留，避免破坏中文排版；其余兼容字符做 NFC 归一
    return unicodedata.normalize("NFC", text.translate(_FULLWIDTH_ALNUM))


def strip_html(text: str) -> str:
    text = _SCRIPT.sub("", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</div>", "\n", text)
    return html.unescape(_TAG.sub("", text))


def strip_markdown_noise(text: str) -> str:
    text = _MD_IMAGE.sub("", text)
    text = _ZHIDA_LINK.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _BOLD.sub(r"\1", text)
    return _ESCAPES.sub(r"\1", text)


def drop_promo_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not _PROMO.search(line))


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = strip_markdown_noise(text)
    text = drop_promo_lines(text)
    lines = [_SPACES.sub(" ", line).rstrip() for line in text.splitlines()]
    return _MULTI_BLANK.sub("\n\n", "\n".join(lines)).strip()


def clean_question(text: str) -> str:
    q = clean_text(text).replace("\n", " ")
    q = _QUESTION_NO.sub("", q).strip()
    # 统一用全角问号结尾（NFKC 会把 ？ 转成 ?，这里再规范回来）
    if q.endswith(("?", "？")) or (q and not q.endswith("。") and _looks_like_question(q)):
        return q.rstrip("?？ ") + "？"
    return q


def _looks_like_question(q: str) -> bool:
    return bool(re.search(r"(什么|为什么|如何|怎么|哪些|区别|原理|是否|吗|呢)", q))


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """对原始题目表做清洗，并剔除题干为空的行。"""
    out = df.copy()
    out["question"] = out["question"].map(clean_question)
    out["answer"] = out["answer"].map(clean_text)
    out["section"] = out["section"].map(lambda s: _QUESTION_NO.sub("", clean_text(s)))
    return out[out["question"].str.len() > 0].reset_index(drop=True)
