-- InterviewAgent PostgreSQL schema (idempotent; executed on every start via spring.sql.init).
-- users 表由 JPA 实体 UserEntity 管理，这里只维护原生 SQL 访问的业务表。

-- 用户长期画像：技能等级 + 薄弱点（JSONB，GIN 索引支持按薄弱知识点反查用户）
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id     VARCHAR(128) PRIMARY KEY,
    name        VARCHAR(256) NOT NULL DEFAULT '',
    skill_level JSONB        NOT NULL DEFAULT '{}'::jsonb,
    weak_points JSONB        NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_user_profiles_weak_points
    ON user_profiles USING GIN (weak_points jsonb_path_ops);

-- 面试记录：结构化评估报告 / 复习计划以 JSONB 存储，GIN 索引支撑多维查询
CREATE TABLE IF NOT EXISTS interview_records (
    id               BIGSERIAL PRIMARY KEY,
    user_id          VARCHAR(128)     NOT NULL,
    session_id       VARCHAR(128)     NOT NULL UNIQUE,
    position         VARCHAR(256)     NOT NULL DEFAULT '',
    overall_score    DOUBLE PRECISION NOT NULL DEFAULT 0,
    overall_level    VARCHAR(8)       NOT NULL DEFAULT '',
    report_json      JSONB            NOT NULL DEFAULT '{}'::jsonb,
    review_plan_json JSONB            NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_interview_records_user_created
    ON interview_records (user_id, created_at DESC);
-- 默认 jsonb_ops：同时支持 @>（包含）与 ?（键/元素存在）查询
CREATE INDEX IF NOT EXISTS idx_interview_records_report
    ON interview_records USING GIN (report_json);

-- 题库：上传题库 & 离线数据管道（ml/）产出的高质量题目统一落库，
-- 作为 BM25 索引的持久化来源，并与 Milvus 中的向量按 id 对齐
CREATE TABLE IF NOT EXISTS question_bank (
    id               VARCHAR(256)     PRIMARY KEY,
    owner_id         VARCHAR(128)     NOT NULL,
    source_file      VARCHAR(256)     NOT NULL DEFAULT '',
    topic            VARCHAR(128)     NOT NULL DEFAULT '',
    question         TEXT             NOT NULL,
    reference_answer TEXT             NOT NULL DEFAULT '',
    question_type    VARCHAR(32)      NOT NULL DEFAULT 'basic',
    difficulty       VARCHAR(16)      NOT NULL DEFAULT 'medium',
    knowledge_points JSONB            NOT NULL DEFAULT '[]'::jsonb,
    quality_score    DOUBLE PRECISION,
    content_hash     VARCHAR(64),
    created_at       TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ      NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_question_bank_owner_source
    ON question_bank (owner_id, source_file);
CREATE INDEX IF NOT EXISTS idx_question_bank_knowledge_points
    ON question_bank USING GIN (knowledge_points jsonb_path_ops);
