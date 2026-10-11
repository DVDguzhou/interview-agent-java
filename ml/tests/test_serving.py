import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from interview_ml.llm import extract_json  # noqa: E402
from interview_ml.serving.app import create_app  # noqa: E402
from interview_ml.serving.resume_parser import merge_extractions  # noqa: E402
from interview_ml.serving.scorer import normalize_score_output  # noqa: E402


class FakeParser:
    def parse(self, **kw):
        return {"name": "张三", "skills": ["Go"], "raw_text": kw["text"], "chunks": 1}


class FakeScorer:
    def score(self, q, a, r):
        return normalize_score_output({"score": 130, "key_points_missed": ["x"]})


def test_resume_endpoint_validates_and_parses():
    client = TestClient(create_app(resume_parser=FakeParser()))
    assert client.post("/v1/resume/parse", json={}).status_code == 400
    resp = client.post("/v1/resume/parse", json={"text": "张三 Go"})
    assert resp.status_code == 200 and resp.json()["skills"] == ["Go"]


def test_score_endpoint_and_unavailable_model():
    ok = TestClient(create_app(scorer_factory=FakeScorer))
    body = ok.post("/v1/score", json={"question": "q", "answer": "a"}).json()
    assert body["score"] == 100 and body["key_points_hit"] == []

    def broken():
        raise RuntimeError("no adapter")

    down = TestClient(create_app(scorer_factory=broken))
    assert down.post("/v1/score", json={"question": "q"}).status_code == 503
    assert down.get("/health").json()["scorer_loaded"] is False


def test_normalize_score_output_follow_up_rule():
    assert normalize_score_output({"score": 55, "key_points_missed": ["a"]})["should_follow_up"] is True
    assert normalize_score_output({"score": 90, "key_points_missed": ["a"]})["should_follow_up"] is False


def test_extract_json_handles_code_fence():
    assert extract_json('结果：```json\n{"score": 70}\n```') == {"score": 70}


def test_merge_extractions_dedups_across_chunks():
    merged = merge_extractions(
        [
            {"name": "张三", "skills": ["Go", "Redis"], "projects": [{"name": "IM 系统", "tech_stack": ["Go"]}]},
            {"skills": ["go", "Kafka"], "projects": [{"name": "IM 系统", "tech_stack": ["Kafka"], "role": "负责人"}]},
        ]
    )
    assert merged["name"] == "张三"
    assert merged["skills"] == ["Go", "Redis", "Kafka"]
    assert len(merged["projects"]) == 1
    assert merged["projects"][0]["tech_stack"] == ["Go", "Kafka"]
    assert merged["projects"][0]["role"] == "负责人"
