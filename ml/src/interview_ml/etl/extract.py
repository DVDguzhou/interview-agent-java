"""结构化抽取：题目 → {topic, knowledge_points, difficulty, question_type, reference_answer}。

默认基于规则（技术词典 + 难度信号词），可选用 LLM 抽取覆盖规则结果（更准但有成本）。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

import pandas as pd

# 技术词典：规范名 → 匹配别名（小写）
TECH_TERMS: dict[str, tuple[str, ...]] = {
    "Go": ("go", "golang", "goroutine", "channel", "gmp", "defer", "slice", "切片"),
    "Java": ("java", "jvm", "spring", "synchronized", "hashmap", "线程池"),
    "Python": ("python", "gil", "asyncio"),
    "MySQL": ("mysql", "innodb", "mvcc", "b+树", "b+ 树", "索引", "事务", "binlog", "redo log", "undo log", "分库分表"),
    "PostgreSQL": ("postgresql", "postgres", "jsonb", "pgvector"),
    "Redis": ("redis", "aof", "rdb", "缓存穿透", "缓存雪崩", "缓存击穿", "跳表", "哨兵"),
    "Kafka": ("kafka", "partition", "消费者组"),
    "RocketMQ": ("rocketmq",),
    "消息队列": ("消息队列", "mq", "消息丢失", "重复消费", "消息积压"),
    "分布式": ("分布式", "raft", "paxos", "cap", "base", "一致性哈希", "分布式锁", "分布式事务", "2pc", "tcc"),
    "微服务": ("微服务", "服务注册", "熔断", "限流", "降级", "网关"),
    "操作系统": ("进程", "线程", "协程", "虚拟内存", "页表", "零拷贝", "epoll"),
    "计算机网络": ("tcp", "udp", "http", "https", "三次握手", "四次挥手", "拥塞控制"),
    "Docker/K8s": ("docker", "k8s", "kubernetes", "容器"),
    "LLM": ("llm", "rag", "embedding", "向量", "agent", "prompt", "微调", "lora", "transformer"),
}

_HARD_SIGNALS = ("源码", "底层", "实现原理", "设计", "如何保证", "高并发", "调度", "一致性", "优化", "架构", "场景")
_EASY_SIGNALS = ("是什么", "有哪些", "区别", "定义", "简述", "介绍一下")
_DESIGN_SIGNALS = ("设计一个", "如何设计", "系统设计", "方案", "场景题")
_EXPERIENCE_SIGNALS = ("你在项目", "你的项目", "遇到过", "实际工作", "线上问题")


def match_terms(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for name, aliases in TECH_TERMS.items():
        for alias in aliases:
            # 纯 ASCII 短别名要求词边界，避免 "go" 命中 "good"
            if alias.isascii() and alias.isalpha():
                if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lower):
                    hits.append(name)
                    break
            elif alias in lower:
                hits.append(name)
                break
    return hits


def infer_difficulty(question: str, answer: str) -> str:
    hard = sum(s in question for s in _HARD_SIGNALS)
    easy = sum(s in question for s in _EASY_SIGNALS)
    if hard >= 1 and len(answer) > 600 or hard >= 2:
        return "hard"
    if easy >= 1 and len(answer) < 400:
        return "easy"
    return "medium"


def infer_type(question: str) -> str:
    if any(s in question for s in _DESIGN_SIGNALS):
        return "design"
    if any(s in question for s in _EXPERIENCE_SIGNALS):
        return "experience"
    return "basic"


def question_id(owner: str, question: str) -> str:
    digest = hashlib.sha1(f"{owner}\n{question}".encode()).hexdigest()[:16]
    return f"etl_{digest}"


def extract_rule_based(row: pd.Series, source_topic: str = "") -> dict:
    question, answer = row["question"], row["answer"]
    kps = match_terms(question) or match_terms(f"{row.get('section', '')} {answer[:300]}")
    topic = kps[0] if kps else (source_topic or row.get("source", ""))
    return {
        "topic": topic,
        "knowledge_points": kps or ([topic] if topic else []),
        "difficulty": infer_difficulty(question, answer),
        "question_type": infer_type(question),
        "reference_answer": answer,
    }


LLM_EXTRACT_SYSTEM = "你是面试题库结构化标注员，只输出 JSON。"
LLM_EXTRACT_USER = """为下面的面试题标注结构化信息。
输出 JSON：{{"topic": "主技术方向", "knowledge_points": ["细粒度知识点，2-5 个"], "difficulty": "easy|medium|hard", "question_type": "basic|experience|design"}}

题目：{question}
参考答案（节选）：{answer}"""


def make_llm_extractor(client) -> Callable[[str, str], dict]:
    def extract(question: str, answer: str) -> dict:
        data = client.chat_json(LLM_EXTRACT_SYSTEM, LLM_EXTRACT_USER.format(question=question, answer=answer[:1500]))
        if data.get("difficulty") not in ("easy", "medium", "hard"):
            data.pop("difficulty", None)
        if data.get("question_type") not in ("basic", "experience", "design"):
            data.pop("question_type", None)
        return {k: v for k, v in data.items() if v}

    return extract


def extract_frame(df: pd.DataFrame, owner: str, llm_extractor: Callable[[str, str], dict] | None = None) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        info = extract_rule_based(row)
        if llm_extractor is not None:
            try:
                info.update(llm_extractor(row["question"], row["answer"]))
            except Exception:  # noqa: BLE001 - 失败回退规则结果
                pass
        records.append(
            {
                "id": question_id(owner, row["question"]),
                "owner_id": owner,
                "source_file": row["source"],
                "question": row["question"],
                "quality_score": float(row.get("quality_score", 0.0)),
                **info,
            }
        )
    return pd.DataFrame(records)
