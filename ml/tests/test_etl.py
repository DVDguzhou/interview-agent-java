from pathlib import Path

import pandas as pd
import pytest

from interview_ml.etl import cleaning, extract, minhash, pipeline, quality, sources

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "questions"


def test_clean_text_removes_html_markdown_and_promo():
    raw = (
        '<div style="x"><h2>关注公众号：XX，回复：面试 领取资料</h2></div>\n'
        "**InnoDB** 支持 [事务](https://example.com/a)　和行锁&nbsp;。\n\n\n\n"
        "![img](https://x/y.png)ＡＢＣ１２３"
    )
    out = cleaning.clean_text(raw)
    assert "公众号" not in out and "<" not in out and "https" not in out
    assert "InnoDB 支持 事务 和行锁" in out
    assert "ABC123" in out  # NFKC 全角转半角
    assert "\n\n\n" not in out


def test_clean_question_strips_numbering():
    assert cleaning.clean_question("**1.3 MySQL中MyISAM与InnoDB的区别是什么?**") == "MySQL中MyISAM与InnoDB的区别是什么？"
    assert cleaning.clean_question("10.2 知道 golang 的内存逃逸吗") == "知道 golang 的内存逃逸吗？"


def test_load_markdown_splits_sections_and_questions(tmp_path):
    md = tmp_path / "redis.md"
    md.write_text("---\ntitle: x\n---\n## 基础\n### 1.1 什么是 AOF？\nAOF 是追加日志。\n### 1.2 RDB？\n快照。\n## 进阶\n### 2.1 集群\n槽位\n")
    df = sources.load_markdown(md)
    assert df["question"].tolist() == ["1.1 什么是 AOF？", "1.2 RDB？", "2.1 集群"]
    assert df["section"].tolist() == ["基础", "基础", "进阶"]
    assert df.loc[0, "answer"] == "AOF 是追加日志。"
    assert set(df["source"]) == {"redis"}


def test_minhash_estimates_jaccard_and_clusters_near_duplicates():
    a = "MySQL 中 InnoDB 和 MyISAM 存储引擎的区别是什么"
    b = "MySQL中InnoDB与MyISAM存储引擎的区别是什么？"
    c = "Redis 的持久化机制 AOF 和 RDB 有什么不同"
    h = minhash.MinHasher(num_perm=256)
    sa, sb = minhash.shingles(a), minhash.shingles(b)
    est = h.jaccard(h.signature(sa), h.signature(sb))
    assert abs(est - minhash.exact_jaccard(sa, sb)) < 0.15
    clusters = minhash.cluster_near_duplicates([a, b, c], threshold=0.6)
    assert clusters[0] == clusters[1] != clusters[2]


def test_optimal_bands_divides_num_perm():
    bands, rows = minhash.optimal_bands(128, 0.8)
    assert bands * rows == 128 and rows > 1


def test_dedup_frame_keeps_longest_answer():
    df = pd.DataFrame(
        {
            "question": ["什么是 MVCC 多版本并发控制", "什么是MVCC多版本并发控制？", "B+ 树为什么适合做索引"],
            "answer": ["短", "这是一个更长更完整的答案", "因为扇出高"],
        }
    )
    out = minhash.dedup_frame(df, threshold=0.7)
    assert len(out) == 2
    assert "这是一个更长更完整的答案" in out["answer"].tolist()


def test_rule_score_flags_low_quality():
    good, reasons = quality.rule_score("Redis 为什么快？", "Redis 基于内存操作，单线程避免了上下文切换，并使用 IO 多路复用模型处理大量连接。")
    assert good >= 0.9 and not reasons
    bad, reasons = quality.rule_score("快？", "略")
    assert bad < 0.6 and "answer_too_short" in reasons


def test_score_frame_calls_llm_only_for_gray_zone():
    df = pd.DataFrame(
        {
            "question": ["Redis 为什么快？", "什么是索引覆盖？", "x"],
            "answer": ["Redis 基于内存操作，单线程避免上下文切换，并使用 IO 多路复用。" * 2, "查询列都在索引中，无需回表", "略"],
        }
    )
    calls = []

    def judge(q, a):
        calls.append(q)
        return 0.9

    out = quality.score_frame(df, llm_judge=judge)
    assert calls == ["什么是索引覆盖？"]
    assert out["keep"].tolist() == [True, True, False]


def test_match_terms_respects_word_boundary():
    assert extract.match_terms("Go 的 GMP 调度模型") == ["Go"]
    assert "Go" not in extract.match_terms("a good answer about mysql")
    assert extract.match_terms("Redis 缓存穿透怎么解决") == ["Redis"]


def test_infer_difficulty_and_type():
    assert extract.infer_difficulty("Redis 是什么？", "内存数据库") == "easy"
    assert extract.infer_difficulty("GMP 调度底层实现原理", "x" * 800) == "hard"
    assert extract.infer_type("如何设计一个秒杀系统") == "design"
    assert extract.infer_type("你在项目中遇到过哪些慢查询") == "experience"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="内置题库不存在")
def test_pipeline_on_bundled_question_bank():
    df, report = pipeline.run([DATA_DIR], owner="public")
    s = report.stages
    assert s["raw"] >= s["cleaned"] >= s["exact_dedup"] >= s["near_dedup"] >= s["quality_kept"] == s["final"] > 50
    assert df["id"].is_unique
    assert set(df["difficulty"]) <= {"easy", "medium", "hard"}
    assert set(df["question_type"]) <= {"basic", "experience", "design"}
    assert (df["owner_id"] == "public").all()
    assert not df["reference_answer"].str.contains("公众号").any()
    assert df["knowledge_points"].map(len).gt(0).mean() > 0.8
