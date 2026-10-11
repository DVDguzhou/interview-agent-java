"""QLoRA 微调评分模型推理：base model（4-bit）+ LoRA adapter，Prompt 与 SFT 数据完全一致。"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..config import Settings, get_settings
from ..llm import extract_json
from ..prompts import build_scorer_messages

log = logging.getLogger("interview_ml.scorer")


def normalize_score_output(data: dict[str, Any]) -> dict[str, Any]:
    score = int(round(max(0.0, min(100.0, float(data.get("score", 50))))))
    missed = [str(x) for x in data.get("key_points_missed") or []]
    return {
        "score": score,
        "feedback": str(data.get("feedback") or ""),
        "key_points_hit": [str(x) for x in data.get("key_points_hit") or []],
        "key_points_missed": missed,
        "should_follow_up": bool(data.get("should_follow_up", 30 <= score < 80 and bool(missed))),
    }


class FineTunedScorer:
    def __init__(self, base_model: str, adapter_path: str, load_in_4bit: bool = True, max_new_tokens: int = 384):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            if load_in_4bit
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base, adapter_path) if adapter_path else base
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self._lock = threading.Lock()  # 单卡推理串行化，避免显存抖动
        self._torch = torch
        log.info("评分模型已加载：base=%s adapter=%s", base_model, adapter_path)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> FineTunedScorer:
        s = settings or get_settings()
        if not s.scorer_adapter_path:
            raise RuntimeError("SCORER_ADAPTER_PATH 未配置")
        return cls(s.scorer_base_model, s.scorer_adapter_path, s.scorer_load_in_4bit)

    def score(self, question: str, answer: str, reference: str) -> dict[str, Any]:
        messages = build_scorer_messages(question, answer, reference)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with self._lock, self._torch.no_grad():
            out = self.model.generate(
                inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(out[0][inputs.shape[-1] :], skip_special_tokens=True)
        return normalize_score_output(extract_json(text))
