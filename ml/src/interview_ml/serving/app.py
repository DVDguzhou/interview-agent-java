"""ML 服务（FastAPI）。

POST /v1/resume/parse   {"text": "..."} 或 {"filename": "cv.pdf", "data": "<base64>"} → 结构化简历
POST /v1/score          {"question", "answer", "reference"} → 评分 JSON（与后端 AnswerScore 字段一致）
GET  /health

评分模型按需懒加载（首次请求时加载到 GPU）；未配置 SCORER_ADAPTER_PATH 时 /v1/score 返回 503，后端自动降级。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__

log = logging.getLogger("interview_ml.app")


class ResumeParseRequest(BaseModel):
    text: str | None = None
    filename: str | None = None
    data: str | None = Field(default=None, description="base64 编码的文件内容")


class ScoreRequest(BaseModel):
    question: str
    answer: str = ""
    reference: str = ""


def create_app(resume_parser=None, scorer_factory=None) -> FastAPI:
    app = FastAPI(title="InterviewAgent ML Service", version=__version__)
    state: dict[str, Any] = {"parser": resume_parser, "scorer": None}
    lock = threading.Lock()

    def get_parser():
        if state["parser"] is None:
            from ..llm import LLMClient
            from .resume_parser import ResumeParser

            state["parser"] = ResumeParser.with_llm(LLMClient())
        return state["parser"]

    def get_scorer():
        if state["scorer"] is None:
            with lock:
                if state["scorer"] is None:
                    if scorer_factory is None:
                        from .scorer import FineTunedScorer

                        state["scorer"] = FineTunedScorer.from_settings()
                    else:
                        state["scorer"] = scorer_factory()
        return state["scorer"]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "scorer_loaded": state["scorer"] is not None}

    @app.post("/v1/resume/parse")
    def parse_resume(req: ResumeParseRequest) -> dict[str, Any]:
        if not req.text and not (req.filename and req.data):
            raise HTTPException(400, "需要提供 text，或 filename + data")
        try:
            return get_parser().parse(text=req.text, filename=req.filename, data_b64=req.data)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/score")
    def score(req: ScoreRequest) -> dict[str, Any]:
        try:
            scorer = get_scorer()
        except Exception as e:  # noqa: BLE001
            log.warning("评分模型不可用: %s", e)
            raise HTTPException(503, f"评分模型不可用: {e}") from e
        return scorer.score(req.question, req.answer, req.reference)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
