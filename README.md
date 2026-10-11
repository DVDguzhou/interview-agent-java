# InterviewAgent · AI 模拟面试平台

一个覆盖 **简历解析 → 智能出题 → 实时面试 → 评估报告** 全流程的 AI 模拟面试平台。后端基于 **Java 21 + Spring Boot 3.4 + [Spring AI Alibaba](https://github.com/alibaba/spring-ai-alibaba)**，采用多 Agent DAG 编排；数据工程与模型微调使用 **Python**（`ml/` 子项目）；前端为 **Next.js + React + TypeScript + Tailwind CSS**（独立仓库 `interview-agent-web`）。

---

## 系统架构

```
浏览器 ──► Nginx (:80)
            ├── /          → Next.js 前端（interview-agent-web，:3000）
            ├── /api/*     → Spring Boot 后端（:9090）REST + SSE 流式输出
            └── /ws        → Spring Boot 后端 WebSocket（面试实时交互）

Spring Boot 后端
  Agent 编排层（Spring AI Alibaba Graph / StateGraph DAG，中心编排器 Orchestrator）
  ┌──────────────────────────────────────────────────────────────────┐
  │ JD 分析 → 简历匹配(结构化简历) → 历史薄弱点召回 → 出题规划(两阶段) │
  │                                   ↓                                │
  │          面试官（多轮问答 + 动态难度调节 + 追问 + 评分）            │
  │                                   ↓                                │
  │          低分题巩固 → 评估报告 → 复习规划（ReactAgent + 工具）      │
  └──────────────────────────────────────────────────────────────────┘
  ┌─────────────────────┬──────────────────────┬─────────────────────┐
  │ RAG 多路召回          │ 记忆系统               │ Skill / 工具         │
  │ Milvus 向量 + BM25    │ 20 条滑窗短期记忆       │ 4 个内置 Skill       │
  │ 去重合并 + 可切换重排  │ Redis + PG 长期画像     │ GitHub / 网页抓取    │
  └─────────────────────┴──────────────────────┴─────────────────────┘
           │                               │
           ▼                               ▼
  ML 服务（ml/，FastAPI :8000）       基础设施（Docker Compose）
  · LlamaIndex 简历解析                PostgreSQL 16（JSONB + GIN）
  · QLoRA 微调评分模型                 Milvus 2.4 · Redis 7
  离线：题库 ETL / SFT 数据集 / 训练评估
```

---

## 核心特性

### 后端与 RAG

- **多 Agent 编排**：7 个 Agent（聊天 / JD 分析 / 简历匹配 / 出题规划 / 面试官 / 评估 / 复习规划）由 `graph/Orchestrator` 基于 StateGraph 编排为有向图，含「未作答即终止」条件分支；面试上下文放在 per-interview 的持有者中，避免 graph state 深拷贝。
- **RAG 多路召回**：Milvus 向量检索（text-embedding-v3，1024 维，COSINE）+ 内存 BM25 双路并行，按文档 ID 去重合并后重排；题库按 `user_id` 做用户级隔离；另实现 RRF（k=60）融合器。
- **可切换重排策略**：`RerankStrategy` 接口 + `Reranker` 分派器，支持 `cross-encoder`（DashScope gte-rerank，默认）/ `llm`（全量重排，只调整顺序不删候选）/ `none`，通过 `RERANKER_TYPE` 切换。
- **RAG 离线评估**：`rag/eval/` 基于标注数据集计算 Recall@K / MRR，输出 Markdown / JSON 报告（`data/eval/reports/`），支持参数 A/B 对比；另有 LLM 三维质量评估（忠实度 / 相关性 / 完整性）。
- **动态难度调节**：两阶段出题（Phase1 规划方向 + 分档候选题池，Phase2 检索 / LLM 出题），`StageScheduler` 按 basic → experience → design 分阶段取题，连续答对升档、连续答错降档。
- **Agent 记忆**：`ChatDispatcher` 维护 20 条滑动窗口短期记忆；长期画像与薄弱点（<60 记录 / ≥80 移除 / 30 天淘汰）走 Redis 缓存 + PostgreSQL 持久化（Cache-Aside）。
- **Skill 系统**：有状态多轮交互的 `Skill` 接口，4 个内置 Skill（快速测验 / 概念教学 / 项目亮点 / 技术对比），`SkillRegistry` 按优先级匹配，WebSocket 与 SSE 共用同一分发逻辑。
- **流式输出（SSE）**：`POST /api/chat/stream` 基于 `ChatModel.stream` + `SseEmitter` 推送增量 token（`delta` / `done` / `error` 事件）。

### 数据建模（PostgreSQL）

表结构见 [`src/main/resources/db/schema.sql`](src/main/resources/db/schema.sql)，启动时幂等执行：

| 表 | 说明 |
|----|------|
| `users` | JPA 实体，bcrypt 密码 |
| `user_profiles` | 长期画像：`skill_level` / `weak_points` 为 JSONB，GIN(`jsonb_path_ops`) 支持按薄弱点反查 |
| `interview_records` | 面试记录：`report_json` / `review_plan_json` 为 JSONB，GIN 索引支撑多维查询；`(user_id, created_at DESC)` 复合索引支撑分页 |
| `question_bank` | 题库：上传题库与离线 ETL 产出统一落库，作为 BM25 持久化来源，id 与 Milvus 向量对齐 |

多维查询接口（`record/InterviewRecordRepository`）把评级 / 薄弱点 / 遗漏知识点等条件转换为 `@>` 包含查询命中 GIN 索引：

```
GET /api/interviews?level=B&weakness=分布式事务&min_score=60&dimension=基础知识&min_dimension_score=70
GET /api/interviews/{sessionId}
GET /api/interviews/stats/weaknesses
```

### 数据工程与模型微调（Python，`ml/`）

- **题库 ETL**：多源采集（Markdown / JSONL / CSV / HTML）→ 清洗（HTML / 噪声 / 推广信息去除、全半角统一）→ 精确去重 + MinHash LSH 近似去重 → 规则 + LLM 质量打分过滤 → 结构化抽取（题目 / 知识点 / 难度 / 参考答案）→ 写入 PostgreSQL 并向量化同步到 Milvus。
- **SFT 数据集**：从 `interview_records` 抽取历史问答，标签对齐（人工分优先）、按分档分层划分训练 / 验证集（防同题泄漏）、生成人工抽检表。
- **QLoRA 微调**：Transformers + PEFT 对 Qwen2.5-7B-Instruct 做 4-bit QLoRA，训练面试回答评分模型；`iml-eval` 输出与人工标注的一致性（Pearson / Spearman / 二次加权 Kappa / MAE）。
- **简历解析**：LlamaIndex 解析 PDF / Word 并分块，Prompt 抽取技能 / 经历 / 项目，作为简历匹配与个性化出题的输入。

后端通过 `ML_SERVICE_URL` 接入 ML 服务：简历解析与微调评分模型均为**可降级**能力，服务不可用时自动回落到通用大模型链路。详见 [`ml/README.md`](ml/README.md)。

---

## 技术栈

| 类别 | 选型 |
|------|------|
| 后端 | Java 21、Spring Boot 3.4、Spring AI Alibaba 1.1（Graph / ReactAgent）、Spring Security + JWT |
| 大模型 | 通义千问 DashScope（qwen-plus）、text-embedding-v3、gte-rerank-v2 |
| 数据 / 模型 | Python、Pandas、NumPy、FastAPI、LlamaIndex、Hugging Face Transformers、PEFT、bitsandbytes |
| 存储 | PostgreSQL 16（JSONB + GIN）、Milvus 2.4、Redis 7 |
| 前端 | Next.js、React、TypeScript、Tailwind CSS、Zustand（仓库 `interview-agent-web`） |
| 工程化 | Docker Compose、Nginx、GitHub Actions（测试 / 镜像发布 / 部署）、Testcontainers |

---

## 项目结构

```
interview-agent-java/
├── src/main/java/com/interview/agent/
│   ├── agent/        # 7 个 Agent（Chat / JDAnalyzer / ResumeMatcher / QuestionPlanner / Interviewer / Evaluator / ReviewPlanner）
│   ├── graph/        # Orchestrator（StateGraph）/ QuestionPool / StageScheduler
│   ├── chat/         # ChatDispatcher（Skill 优先 + 短期记忆）/ ChatStreamController（SSE）
│   ├── rag/          # MilvusStore / BM25 / QuestionBankStore / 重排策略 / eval 离线评估
│   ├── memory/       # 短期记忆 / 长期画像 / RedisStore / PostgresStore / CombinedStore
│   ├── record/       # 面试记录多维查询 Repository + REST Controller
│   ├── ml/           # MlServiceClient（简历解析 / 微调评分，失败降级）
│   ├── skill/        # Skill 接口、注册中心与 4 个内置 Skill
│   ├── loader/       # PDF / DOCX / Web / 题库解析
│   ├── mcp/          # GitHub 搜索 / 网页抓取工具
│   ├── auth/ config/ handler/ model/
├── src/main/resources/{application.yml, db/schema.sql}
├── ml/                     # Python：ETL / SFT / QLoRA / FastAPI 服务
├── data/questions/         # 内置题库（Go / MySQL / 分布式 / MQ / Redis）
├── data/eval/              # RAG 评估数据集、manifest 与评估报告
├── deploy/nginx.conf       # 统一入口反向代理（含 SSE / WebSocket 配置）
├── docker-compose.yml      # 基础设施 + 全栈（--profile app）
└── .github/workflows/ci.yml
```

---

## 快速开始

### 1. 环境准备

| 工具 | 版本 | 说明 |
|------|------|------|
| JDK | 21 | 更高版本会因 Lombok 注解处理器不兼容导致编译失败（多 JDK 环境：`export JAVA_HOME=$(/usr/libexec/java_home -v 21)`） |
| Maven | 3.9+ | |
| Docker | 最新 | 基础设施 / 集成测试 |
| Node.js | 20.9+ | 前端 |
| uv | 最新 | ML 子项目（可选） |

DashScope API Key：[DashScope 控制台](https://dashscope.console.aliyun.com/) → API-KEY 管理。

### 2. 配置与启动

```bash
cp .env.example .env          # 填入 DASHSCOPE_API_KEY，其余默认值与 docker-compose 对应
make infra-up                 # Milvus + Redis + PostgreSQL
make run                      # 后端 :9090（spring-dotenv 自动加载 .env）

cd ../interview-agent-web && npm install && npm run dev   # 前端 :3000
```

可选：启动 ML 服务并在 `.env` 中设置 `ML_SERVICE_URL=http://localhost:8000`：

```bash
cd ml && uv sync --extra serve && uv run iml-serve
```

### 3. 全栈部署

```bash
docker compose --profile app up -d --build    # backend + ml + web + nginx，访问 http://<host>/
```

CI（`.github/workflows/ci.yml`）在每次 push 时运行后端测试（含 Testcontainers PostgreSQL）与 ML 测试，main 分支推送镜像到 GHCR；打 `v*` tag 且配置了 `DEPLOY_*` Secrets 时通过 SSH 拉取最新镜像完成部署。

### 4. 使用

1. 注册 / 登录。
2. （可选）上传题库：PDF / TXT / MD，LLM 解析后写入 PostgreSQL 题库 + Milvus + BM25（按用户隔离）；也可用 `make ml-etl OWNER=<用户名>` 批量导入 `data/questions/`。
3. 开始面试：填入 JD（文本 / 文件 / 链接）与简历，系统依次执行 JD 分析 → 简历匹配 → 题库检索 → 出题规划 → 逐题面试 → 评估报告 → 复习计划。
4. 在「历史评估报告」页查看分数趋势、维度雷达图、高频薄弱点，并按评级 / 薄弱点 / 分数区间筛选。

---

## 常用命令

```bash
make run / build / test / package      # 后端
make infra-up / infra-down / infra-status
make deploy                            # 全栈 docker compose
make ml-test / ml-etl                  # Python 子项目
```

RAG 离线评估：带命令行参数启动后端（`--prepare` / `--gen-dataset` / `--execute`，见 `rag/eval/EvalCommandRunner`），报告输出到 `data/eval/reports/`。

---

## 常见问题

**启动报 `Could not resolve placeholder 'DASHSCOPE_API_KEY'` 或 LLM 401**：确认 `.env` 中填写了真实 Key。

**`mvn` 编译报 `ExceptionInInitializerError ... TypeTag :: UNKNOWN`**：JDK 版本过高，切换到 JDK 21。

**集成测试被跳过**：`PostgresPersistenceTest` 依赖 Docker；macOS Docker Desktop 若未创建 `/var/run/docker.sock`，执行 `export DOCKER_HOST=unix://$HOME/.docker/run/docker.sock`。

**Milvus 启动后 collection 加载卡住**：Milvus standalone 重启后元数据残留，清空数据卷后重启（会清空已上传题库的向量，PostgreSQL 题库保留，可用 ETL 重新导入）：

```bash
docker compose down && docker volume rm interview-agent-java_milvus_data && docker compose up -d
```

**SSE 输出不是逐字出现**：确认反向代理关闭了缓冲（见 `deploy/nginx.conf` 中 `/api/chat/stream`）。

---

## 致谢

本项目在开源项目「AI 模拟面试官」（作者：公众号「IT杨秀才」，[golangstar.cn](https://golangstar.cn/)）的基础上二次开发，`data/questions/` 下的内置题库内容同样来源于该作者。在原项目基础上新增 / 重构了 PostgreSQL 数据层（JSONB + GIN 多维查询、题库持久化）、SSE 流式对话、Python 数据工程与 QLoRA 微调子项目、ML 服务接入、Next.js 前端与报告可视化、容器化部署与 CI/CD。感谢原作者的开源分享。
