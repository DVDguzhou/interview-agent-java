package com.interview.agent.rag;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * 题库持久化（PostgreSQL {@code question_bank} 表）。
 * <p>
 * 用户上传的题库与离线数据管道（ml/）清洗产出的题目都写入这里：
 * <ul>
 *   <li>作为 BM25 内存索引的持久化来源，服务重启后按用户懒加载重建；</li>
 *   <li>id 与 Milvus 中的向量记录一一对应，便于双路召回结果按文档 ID 去重合并。</li>
 * </ul>
 */
@Slf4j
@Component
public class QuestionBankStore {

    private final NamedParameterJdbcTemplate jdbc;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public QuestionBankStore(NamedParameterJdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @Builder
    public record QuestionRow(String id, String ownerId, String sourceFile, String topic, String question,
                              String referenceAnswer, String questionType, String difficulty,
                              List<String> knowledgePoints, Double qualityScore) {
    }

    /** 覆盖写入某用户某来源文件的题目（先删后插；行上的 owner / source 以入参为准） */
    public void replaceSourceFile(String ownerId, String sourceFile, List<QuestionRow> rows) {
        jdbc.update("DELETE FROM question_bank WHERE owner_id = :owner AND source_file = :source",
                Map.of("owner", ownerId, "source", sourceFile));
        upsert(rows.stream().map(r -> new QuestionRow(r.id(), ownerId, sourceFile, r.topic(), r.question(),
                r.referenceAnswer(), r.questionType(), r.difficulty(), r.knowledgePoints(), r.qualityScore())).toList());
    }

    public void upsert(List<QuestionRow> rows) {
        if (rows == null || rows.isEmpty()) return;
        SqlParameterSource[] batch = rows.stream().map(r -> new MapSqlParameterSource()
                .addValue("id", r.id())
                .addValue("owner", r.ownerId())
                .addValue("source", nz(r.sourceFile()))
                .addValue("topic", nz(r.topic()))
                .addValue("question", r.question())
                .addValue("reference", nz(r.referenceAnswer()))
                .addValue("type", r.questionType() != null ? r.questionType() : "basic")
                .addValue("difficulty", r.difficulty() != null ? r.difficulty() : "medium")
                .addValue("kps", toJson(r.knowledgePoints()))
                .addValue("quality", r.qualityScore())).toArray(SqlParameterSource[]::new);
        jdbc.batchUpdate("""
                INSERT INTO question_bank (id, owner_id, source_file, topic, question, reference_answer,
                                           question_type, difficulty, knowledge_points, quality_score)
                VALUES (:id, :owner, :source, :topic, :question, :reference, :type, :difficulty,
                        CAST(:kps AS jsonb), :quality)
                ON CONFLICT (id) DO UPDATE SET
                    owner_id = EXCLUDED.owner_id, source_file = EXCLUDED.source_file, topic = EXCLUDED.topic,
                    question = EXCLUDED.question, reference_answer = EXCLUDED.reference_answer,
                    question_type = EXCLUDED.question_type, difficulty = EXCLUDED.difficulty,
                    knowledge_points = EXCLUDED.knowledge_points, quality_score = EXCLUDED.quality_score,
                    updated_at = now()
                """, batch);
    }

    /** 读取用户题库，转换为 BM25 文档（content 格式与 Milvus 写入保持一致） */
    public List<RagDocument> loadDocuments(String ownerId) {
        return jdbc.query("""
                SELECT id, question, reference_answer, question_type, difficulty, knowledge_points::text AS kps
                FROM question_bank WHERE owner_id = :owner
                """, Map.of("owner", ownerId), (rs, i) -> RagDocument.builder()
                .id(rs.getString("id"))
                .content(rs.getString("question") + "\n参考答案：" + rs.getString("reference_answer"))
                .userId(ownerId)
                .metadata(Map.of(
                        "type", rs.getString("question_type"),
                        "difficulty", rs.getString("difficulty"),
                        "skills", readList(rs.getString("kps"))))
                .build());
    }

    private List<String> readList(String json) {
        try {
            return objectMapper.readValue(json, new TypeReference<List<String>>() {});
        } catch (Exception e) {
            return List.of();
        }
    }

    private String toJson(List<String> list) {
        try {
            return objectMapper.writeValueAsString(list != null ? list : List.of());
        } catch (Exception e) {
            return "[]";
        }
    }

    private static String nz(String s) {
        return s != null ? s : "";
    }
}
