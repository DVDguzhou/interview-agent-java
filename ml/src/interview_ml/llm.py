"""OpenAI 兼容协议的轻量 LLM / Embedding 客户端（默认对接 DashScope compatible-mode）。"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

import httpx

from .config import Settings, get_settings

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出中提取第一个 { 到最后一个 } 之间的 JSON（兼容 ```json 包裹）。"""
    m = _JSON_BLOCK.search(text or "")
    if not m:
        raise ValueError(f"no json object in output: {text[:200]!r}")
    return json.loads(m.group(0))


class LLMClient:
    def __init__(self, settings: Settings | None = None, timeout: float = 120.0, max_retries: int = 3):
        self.settings = settings or get_settings()
        if not self.settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置")
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=self.settings.llm_base_url,
            headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}"},
            timeout=timeout,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._http.post(path, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM 请求失败: {last}")

    def chat(self, system: str, user: str, temperature: float = 0.1, model: str | None = None) -> str:
        data = self._post(
            "/chat/completions",
            {
                "model": model or self.settings.llm_model,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        return data["choices"][0]["message"]["content"]

    def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
        return extract_json(self.chat(system, user, **kw))

    def embed(self, texts: Sequence[str], batch_size: int = 10) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            data = self._post(
                "/embeddings",
                {
                    "model": self.settings.embedding_model,
                    "input": list(texts[i : i + batch_size]),
                    "dimensions": self.settings.embedding_dim,
                },
            )
            out.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        return out
