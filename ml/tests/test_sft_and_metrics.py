import json

import pandas as pd
import pytest

from interview_ml.sft import build_dataset as sft
from interview_ml.train import metrics


def _raw(n=60):
    rows = []
    for i in range(n):
        rows.append(
            {
                "question": f"问题 {i % 20}",
                "answer": f"回答 {i}" if i % 10 else "不会",
                "score": (i * 7) % 110 - 5,  # 含越界值
                "feedback": "fb",
                "key_points_hit": ["a"],
                "key_points_missed": ["b"] if i % 2 else [],
                "reference": "ref",
            }
        )
    return pd.DataFrame(rows)


def test_align_labels_clips_and_prefers_human_scores():
    raw = _raw()
    human = pd.DataFrame([{"question": "问题 1", "answer": "回答 1", "human_score": 88}])
    out = sft.align_labels(raw, human)
    assert out["score"].between(0, 100).all()
    row = out[(out["question"] == "问题 1") & (out["answer"] == "回答 1")].iloc[0]
    assert row["score"] == 88 and row["label_source"] == "human"
    assert (out.loc[out["answer"] == "不会", "score"] <= 10).all()
    assert set(out["score_bucket"]) <= set(sft.BUCKET_LABELS)


def test_stratified_split_has_no_question_leakage():
    data = sft.align_labels(_raw(200))
    train, val = sft.stratified_split(data, val_ratio=0.2)
    assert len(train) + len(val) == len(data) and len(val) > 0
    assert not set(train["question"]) & set(val["question"])


def test_to_messages_matches_scorer_prompt_format():
    data = sft.align_labels(_raw(10))
    ex = sft.to_messages(data.iloc[1])
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    target = json.loads(ex["messages"][-1]["content"])
    assert target["score"] == ex["score"]
    assert set(target) == {"score", "feedback", "key_points_hit", "key_points_missed", "should_follow_up"}


def test_agreement_metrics():
    human = [10, 35, 55, 75, 95, 60, 20, 85]
    assert metrics.agreement_report(human, human)["qwk"] == pytest.approx(1.0)
    assert metrics.spearman(human, human) == pytest.approx(1.0)
    noisy = [h + 8 for h in human]
    rep = metrics.agreement_report(noisy, human)
    assert rep["pearson"] == pytest.approx(1.0) and rep["mae"] == 8 and rep["within_10"] == 1.0
    reversed_ = [100 - h for h in human]
    assert metrics.agreement_report(reversed_, human)["qwk"] < 0
    assert metrics.to_buckets([0, 29, 30, 69, 70, 100]).tolist() == [0, 0, 1, 2, 3, 4]
