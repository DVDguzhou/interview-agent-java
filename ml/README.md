# interview-ml：数据工程与模型服务

InterviewAgent 的 Python 子项目，负责三件事：

1. **题库 ETL**：多源面试题采集 → 清洗 → MinHash 近似去重 → 规则 + LLM 质量过滤 → 结构化抽取 → 写入 PostgreSQL `question_bank` 并向量化写入 Milvus（与后端 `MilvusStore` 同构，按 `owner_id` 做用户级隔离）。
2. **评分模型微调**：把历史面试问答整理成 SFT 数据集，基于 Transformers + PEFT 对 Qwen2.5-7B-Instruct 做 QLoRA（4-bit）微调，并与人工标注做一致性评估。
3. **在线服务（FastAPI）**：LlamaIndex 简历解析 `/v1/resume/parse`、微调评分模型推理 `/v1/score`，后端通过 `ML_SERVICE_URL` 接入，不可用时自动降级到通用大模型。

## 目录

```
ml/
├── src/interview_ml/
│   ├── etl/            sources（md/jsonl/csv/html）· cleaning · minhash · quality · extract · sinks · pipeline(CLI)
│   ├── sft/            build_dataset：标签对齐、分层划分（防题目泄漏）、人工抽检表
│   ├── train/          qlora：QLoRA 训练 · evaluate：一致性评估 · metrics：Pearson/Spearman/QWK/MAE
│   ├── serving/        app(FastAPI) · resume_parser(LlamaIndex) · scorer(base + LoRA adapter)
│   ├── prompts.py      评分 Prompt（训练 / 推理共用，保证分布一致）
│   └── llm.py          OpenAI 兼容 LLM / Embedding 客户端（默认 DashScope）
├── configs/qlora_qwen2.5_7b.yaml
└── tests/
```

## 快速开始

```bash
cd ml
uv sync --group dev                 # 核心依赖 + 测试
uv run pytest                       # 单元测试（含对内置题库跑一遍完整 ETL）
```

可选依赖：`uv sync --extra milvus`（写 Milvus）、`--extra serve`（FastAPI + LlamaIndex）、`--extra train`（GPU 训练）。

## 1. 题库 ETL

```bash
# 仅规则流程：输出 build/etl/questions.jsonl + report.json（各阶段条数、丢弃原因、难度 / 主题分布）
uv run iml-etl --input ../data/questions --owner alice

# 全流程：LLM 灰区打分 + LLM 结构化抽取 + 落库（需 DASHSCOPE_API_KEY、PG_DSN、MILVUS_URI）
uv run --extra milvus iml-etl --input raw/ --owner alice --llm-judge --llm-extract --to-postgres --to-milvus
```

| 阶段 | 实现 |
|------|------|
| 清洗 | HTML 标签 / 图片 / 链接 / 转义符去除，NFKC 全半角归一，推广引流行过滤，题号剥离 |
| 去重 | 规范化题干 SHA-256 精确去重 → 字符 3-gram MinHash（128 perm）+ LSH 分桶，并查集聚簇，每簇保留答案最完整的一条 |
| 质量过滤 | 规则分（长度、占位符、代码块闭合、中文占比、答案复读题干）；规则分落在灰区 [0.4, 0.7) 的样本再交给 LLM 打分，最终分 = 0.4·规则 + 0.6·LLM |
| 结构化抽取 | 技术词典匹配知识点、信号词推断难度与题型；`--llm-extract` 用 LLM 覆盖 |
| 落库 | PostgreSQL `question_bank`（schema 复用后端 `db/schema.sql`）+ Milvus `interview_questions`（1024 维，text-embedding-v3） |

导入后，后端 BM25 会在该用户首次检索时从 `question_bank` 懒加载，Milvus 侧按 `user_id` 过滤召回。

## 2. 评分模型微调

```bash
# ① 构建 SFT 数据集：从线上面试记录（interview_records.report_json.detailed_review）抽取问答
uv run iml-build-sft --from-postgres --human-labels labels.csv --out build/sft
#    产出 train.jsonl / val.jsonl（chat messages）、manual_review.csv（按分档分层抽样的人工抽检表）、stats.json

# ② QLoRA 训练（单卡 24GB）
uv run --extra train iml-train --config configs/qlora_qwen2.5_7b.yaml

# ③ 一致性评估：通用大模型基线 vs 微调模型，对比人工标注
SCORER_ADAPTER_PATH=build/qlora-scorer/adapter \
uv run --extra train iml-eval --labels build/sft/human_eval.jsonl --scorer llm --scorer finetuned
#    输出 build/eval/agreement.json：pearson / spearman / qwk（二次加权 Kappa）/ mae / within_10 / bucket_accuracy
```

要点：只对 assistant 段计算 loss；按题目分层划分训练 / 验证集，避免同一道题泄漏到验证集；人工标注分优先覆盖模型分。

## 3. 在线服务

```bash
uv run --extra serve iml-serve                      # :8000
SCORER_ADAPTER_PATH=build/qlora-scorer/adapter \
uv run --extra serve --extra train iml-serve        # 同时启用微调评分模型（需 GPU）
```

后端配置：`ML_SERVICE_URL=http://localhost:8000`，`ML_SCORER_ENABLED=true` 开启微调评分（默认关闭，仅启用简历解析）。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | — | LLM / Embedding |
| `LLM_BASE_URL` | DashScope compatible-mode | 任意 OpenAI 兼容端点 |
| `PG_DSN` | `postgresql+psycopg://interview:interview@localhost:5432/interview_agent` | |
| `MILVUS_URI` | `http://localhost:19530` | |
| `SCORER_BASE_MODEL` / `SCORER_ADAPTER_PATH` | `Qwen/Qwen2.5-7B-Instruct` / 空 | 评分模型 |
