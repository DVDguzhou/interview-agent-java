"""MinHash + LSH 近似去重（纯 NumPy 实现）。

- 文本 → 字符 n-gram shingle 集合（中文场景下字符级比词级更稳健）
- 对每个 shingle 做 num_perm 个独立的通用哈希 h_i(x) = (a_i * x + b_i) mod p，取最小值得到签名
- LSH：签名切成 bands × rows，任一 band 完全相同即成为候选对，再用签名估计的 Jaccard 复核
- 候选对用并查集合并成簇，每簇保留一个代表（默认保留答案最长、信息量最大的一条）
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
_NON_WORD = re.compile(r"[\s\W_]+", re.U)


def shingles(text: str, n: int = 3) -> set[int]:
    norm = _NON_WORD.sub("", text.lower())
    if len(norm) < n:
        return {_hash32(norm)} if norm else set()
    return {_hash32(norm[i : i + n]) for i in range(len(norm) - n + 1)}


def _hash32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(_MERSENNE_PRIME), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE_PRIME), size=num_perm, dtype=np.uint64)

    def signature(self, shingle_set: set[int]) -> np.ndarray:
        if not shingle_set:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        x = np.fromiter(shingle_set, dtype=np.uint64)[:, None]
        # uint64 乘法溢出按 2^64 取模，再对梅森素数取模，与 datasketch 的做法一致
        hv = ((x * self.a + self.b) % _MERSENNE_PRIME) & _MAX_HASH
        return hv.min(axis=0)

    @staticmethod
    def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


def exact_jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def optimal_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """选择使 S 曲线拐点 (1/b)^(1/r) 最接近阈值的 (bands, rows)。"""
    best = (num_perm, 1)
    best_err = float("inf")
    for rows in range(1, num_perm + 1):
        if num_perm % rows:
            continue
        bands = num_perm // rows
        err = abs((1 / bands) ** (1 / rows) - threshold)
        if err < best_err:
            best, best_err = (bands, rows), err
    return best


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_near_duplicates(
    texts: Sequence[str], threshold: float = 0.8, num_perm: int = 128, ngram: int = 3, seed: int = 42
) -> list[int]:
    """返回每条文本所属簇的代表下标（簇内最小下标）。"""
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    sigs = np.stack([hasher.signature(shingles(t, ngram)) for t in texts]) if texts else np.empty((0, num_perm))
    bands, rows = optimal_bands(num_perm, threshold)
    uf = _UnionFind(len(texts))

    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        chunk = sigs[:, band * rows : (band + 1) * rows]
        for idx in range(len(texts)):
            buckets[chunk[idx].tobytes()].append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            # 小桶两两复核；超大桶（多为模板化短文本）只与桶首比较，避免 O(n^2)
            pairs = (
                ((members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members)))
                if len(members) <= 64
                else ((members[0], m) for m in members[1:])
            )
            for a, b in pairs:
                if uf.find(a) != uf.find(b) and MinHasher.jaccard(sigs[a], sigs[b]) >= threshold:
                    uf.union(a, b)
    return [uf.find(i) for i in range(len(texts))]


def dedup_frame(
    df: pd.DataFrame,
    text_col: str = "question",
    threshold: float = 0.8,
    num_perm: int = 128,
    prefer_col: str | None = "answer",
) -> pd.DataFrame:
    """对 DataFrame 做近似去重，新增 dup_cluster 列；每簇保留 prefer_col 最长的一条。"""
    if df.empty:
        return df.assign(dup_cluster=pd.Series(dtype=int))
    out = df.reset_index(drop=True).copy()
    out["dup_cluster"] = cluster_near_duplicates(out[text_col].tolist(), threshold=threshold, num_perm=num_perm)
    if prefer_col:
        order = out[prefer_col].str.len().rank(method="first", ascending=False)
        out = out.assign(_rank=order).sort_values(["dup_cluster", "_rank"]).drop(columns="_rank")
    return out.drop_duplicates("dup_cluster", keep="first").sort_index().reset_index(drop=True)
