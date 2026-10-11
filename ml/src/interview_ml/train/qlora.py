"""基于 Hugging Face Transformers + PEFT 的 QLoRA（4-bit NF4）微调。

用法：
    iml-train --config configs/qlora_qwen2.5_7b.yaml
单卡 24GB（如 A10 / 4090）可跑 Qwen2.5-7B-Instruct，seq_len 2048、batch 2 × grad_accum 8。
只对 assistant 段计算 loss（prompt 部分 label 置为 -100），避免模型去拟合题干与回答文本。
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("interview_ml.train")

IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    train_file: str = "build/sft/train.jsonl"
    val_file: str = "build/sft/val.jsonl"
    output_dir: str = "build/qlora-scorer"
    max_seq_len: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    learning_rate: float = 2e-4
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> TrainConfig:
        return cls(**(yaml.safe_load(path.read_text(encoding="utf-8")) or {}))


def tokenize_example(example: dict, tokenizer, max_len: int) -> dict:
    """chat template 编码，prompt 部分 mask 掉，只监督 assistant 输出。"""
    messages = example["messages"]
    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    full_ids = full_ids[:max_len]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="QLoRA 微调面试回答评分模型")
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig.from_yaml(args.config)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": cfg.train_file, "validation": cfg.val_file})
    ds = ds.map(
        lambda ex: tokenize_example(ex, tokenizer, cfg.max_seq_len),
        remove_columns=ds["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
    )
    trainer.train()

    adapter_dir = Path(cfg.output_dir) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "train_config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")
    log.info("LoRA adapter 已保存到 %s", adapter_dir)


if __name__ == "__main__":
    main()
